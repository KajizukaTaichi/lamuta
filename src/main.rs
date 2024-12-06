use std::{
    collections::BTreeMap,
    env::{self, args},
    fs::read_to_string,
    io::{self, Write},
    path::Path,
};

const VERSION: &str = "0.1.0";
const SPACE: [char; 5] = [' ', '　', '\n', '\t', '\r'];

fn main() {
    let args: Vec<String> = args().collect();
    if let Some(path) = args.get(1) {
        if let Ok(code) = read_to_string(path) {
            if let Some(parent_dir) = Path::new(path).parent() {
                env::set_current_dir(parent_dir).unwrap_or_default();
            }

            if let Some(ast) = Engine::parse(code) {
                let mut engine = Engine::new();
                if engine.eval(ast).is_none() {
                    println!("Error! something is wrong at runtime")
                }
            } else {
                println!("Error! something is wrong at syntax")
            }
        } else {
            eprintln!("Error! the file can't be opened")
        }
    } else {
        println!("Lamuta {VERSION}");
        let mut engine = Engine::new();
        loop {
            print!("> ");
            io::stdout().flush().unwrap();
            let mut buffer = String::new();
            if io::stdin().read_line(&mut buffer).is_ok() {
                let buffer = buffer.trim().to_string();
                if buffer == ":q" {
                    break;
                }
                if let Some(ast) = Engine::parse(buffer) {
                    if let Some(result) = engine.eval(ast) {
                        println!("{}", result.get_symbol())
                    }
                }
            }
        }
    }
}

type Scope = BTreeMap<String, Type>;
type Program = Vec<Statement>;
#[derive(Debug, Clone)]
struct Engine {
    scope: Scope,
}

impl Engine {
    fn new() -> Engine {
        Engine {
            scope: BTreeMap::from([
                (
                    "type".to_string(),
                    Type::Function(Function::BuiltIn(|expr, _| {
                        Some(Type::Text(
                            match expr {
                                Type::Number(_) => "number",
                                Type::Text(_) => "text",
                                Type::Symbol(_) => "symbol",
                                Type::List(_) => "list",
                                Type::Function(_) => "function",
                                Type::Null => "null",
                            }
                            .to_string(),
                        ))
                    })),
                ),
                (
                    "load".to_string(),
                    Type::Function(Function::BuiltIn(|expr, engine| {
                        if let Ok(module) = read_to_string(expr.get_text()) {
                            let module = Engine::parse(module)?;
                            Some(engine.eval(module)?)
                        } else {
                            None
                        }
                    })),
                ),
                (
                    "input".to_string(),
                    Type::Function(Function::BuiltIn(|expr, _| {
                        let prompt = expr.get_text();
                        print!("{prompt}");
                        io::stdout().flush().unwrap();
                        let mut buffer = String::new();
                        if io::stdin().read_line(&mut buffer).is_ok() {
                            Some(Type::Text(buffer.trim().to_string()))
                        } else {
                            None
                        }
                    })),
                ),
                (
                    "range".to_string(),
                    Type::Function(Function::BuiltIn(|params, _| {
                        let params = params.get_list();
                        if params.len() == 1 {
                            let mut range: Vec<Type> = vec![];
                            let mut current: f64 = 0.0;
                            while current < params[0].get_number()? {
                                range.push(Type::Number(current));
                                current += 1.0;
                            }
                            Some(Type::List(range))
                        } else if params.len() == 2 {
                            let mut range: Vec<Type> = vec![];
                            let mut current: f64 = params[0].get_number()?;
                            while current < params[1].get_number()? {
                                range.push(Type::Number(current));
                                current += 1.0;
                            }
                            Some(Type::List(range))
                        } else if params.len() == 3 {
                            let mut range: Vec<Type> = vec![];
                            let mut current: f64 = params[0].get_number()?;
                            while current < params[1].get_number()? {
                                range.push(Type::Number(current));
                                current += params[2].get_number()?;
                            }
                            Some(Type::List(range))
                        } else {
                            None
                        }
                    })),
                ),
                ("new-line".to_string(), Type::Text("\n".to_string())),
                ("carriage-return".to_string(), Type::Text("\r".to_string())),
                ("double-quote".to_string(), Type::Text("\"".to_string())),
                ("space".to_string(), Type::Text(" ".to_string())),
                ("tab".to_string(), Type::Text("\t".to_string())),
            ]),
        }
    }

    fn parse(source: String) -> Option<Program> {
        let mut program: Program = Vec::new();
        for line in tokenize(source, vec![';'])? {
            let line = line.trim().to_string();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            program.push(Statement::parse(line.trim().to_string())?);
        }
        Some(program)
    }

    fn eval(&mut self, program: Program) -> Option<Type> {
        let mut result = Type::Null;
        for code in program {
            result = match code {
                Statement::Value(expr) => expr.eval(self)?,
                Statement::Print(expr) => {
                    for i in expr {
                        print!(
                            "{}",
                            match i.eval(self)? {
                                Type::Text(text) => text,
                                other => other.get_symbol(),
                            }
                        );
                    }
                    io::stdout().flush().unwrap();
                    Type::Null
                }
                Statement::Let(name, expr) => {
                    let val = expr.eval(self)?;
                    if name != "_" {
                        self.scope.insert(name, val.clone());
                    }
                    Type::Null
                }
                Statement::If(expr, then, r#else) => {
                    if let Some(it) = expr.eval(self) {
                        self.scope.insert("it".to_string(), it);
                        then.eval(self)?
                    } else {
                        if let Some(r#else) = r#else {
                            r#else.eval(self)?
                        } else {
                            Type::Null
                        }
                    }
                }
                Statement::Match(expr, conds) => {
                    let expr = expr.eval(self)?;
                    let mut result = Type::Null;
                    'top: for (conds, value) in conds {
                        for cond in conds {
                            let cond = cond.eval(self)?;
                            if expr.is_match(&cond) {
                                result = value.eval(self)?;
                                break 'top;
                            }
                        }
                    }
                    result
                }
                Statement::While(expr, code) => {
                    let mut result = Type::Null;
                    while let Some(it) = expr.eval(self) {
                        self.scope.insert("it".to_string(), it);
                        result = code.eval(self)?;
                    }
                    result
                }
                Statement::For(counter, expr, code) => {
                    let mut result = Type::Null;
                    for i in expr.eval(self)?.get_list() {
                        if counter != "_" {
                            self.scope.insert(counter.clone(), i);
                        }
                        result = code.eval(self)?;
                    }
                    result
                }
                Statement::Fault => return None,
            };
        }
        Some(result)
    }
}

#[derive(Debug, Clone)]
enum Statement {
    Value(Expr),
    Print(Vec<Expr>),
    Let(String, Expr),
    If(Expr, Expr, Option<Expr>),
    Match(Expr, Vec<(Vec<Expr>, Expr)>),
    For(String, Expr, Expr),
    While(Expr, Expr),
    Fault,
}

impl Statement {
    fn parse(code: String) -> Option<Statement> {
        let code = code.trim();
        if code.starts_with("(") && code.ends_with(")") {
            let code = {
                let mut code = code.to_string();
                code.remove(0);
                code.remove(code.len() - 1);
                code
            };
            Some(Statement::Value(Expr::parse(code)?))
        } else if code.starts_with("print") {
            let mut exprs = vec![];
            for i in tokenize(code["print".len()..].to_string(), vec![','])? {
                exprs.push(Expr::parse(i)?)
            }
            Some(Statement::Print(exprs))
        } else if code.starts_with("if") {
            let code = code["if".len()..].to_string();
            let code = tokenize(code, SPACE.to_vec())?;
            if code.get(2).and_then(|x| Some(x == "else")).unwrap_or(false) {
                Some(Statement::If(
                    Expr::parse(code.get(0)?.to_string())?,
                    Expr::parse(code.get(1)?.to_string())?,
                    Some(Expr::parse(code.get(3)?.to_string())?),
                ))
            } else {
                Some(Statement::If(
                    Expr::parse(code.get(0)?.to_string())?,
                    Expr::parse(code.get(1)?.to_string())?,
                    None,
                ))
            }
        } else if code.starts_with("match") {
            let code = code["match".len()..].to_string();
            let tokens = tokenize(code, SPACE.to_vec())?;
            let expr = Expr::parse(tokens.get(0)?.to_string())?;
            let tokens = tokenize(
                tokens.get(1)?[1..tokens.get(1)?.len() - 1].to_string(),
                vec![','],
            )?;
            let mut conds = vec![];
            for i in tokens {
                let tokens = tokenize(i, vec!['='])?;
                let mut cond = vec![];
                for i in tokenize(tokens.get(0)?.to_string(), vec!['|'])? {
                    cond.push(Expr::parse(i.to_string())?)
                }
                conds.push((cond, Expr::parse(tokens.get(1)?.to_string())?))
            }
            Some(Statement::Match(expr, conds))
        } else if code.starts_with("for") {
            let code = code["for".len()..].to_string();
            let code = tokenize(code, SPACE.to_vec())?;
            if code.get(1).and_then(|x| Some(x == "in")).unwrap_or(false) {
                Some(Statement::For(
                    code.get(0)?.to_string(),
                    Expr::parse(code.get(2)?.to_string())?,
                    Expr::parse(code.get(3)?.to_string())?,
                ))
            } else {
                None
            }
        } else if code.starts_with("while") {
            let code = code["while".len()..].to_string();
            let code = tokenize(code, SPACE.to_vec())?;
            Some(Statement::While(
                Expr::parse(code.get(0)?.to_string())?,
                Expr::parse(code.get(1)?.to_string())?,
            ))
        } else if code.starts_with("let") {
            let code = code["let".len()..].to_string();
            let (name, code) = code.split_once("=")?;
            Some(Statement::Let(
                name.trim().to_string(),
                Expr::parse(code.to_string())?,
            ))
        } else if code == "fault" {
            Some(Statement::Fault)
        } else {
            Some(Statement::Value(Expr::parse(code.to_string())?))
        }
    }

    fn reparse(&self) -> String {
        match self {
            Statement::Value(expr) => format!("{}", expr.reparse()),
            Statement::If(cond, then, r#else) => {
                if let Some(r#else) = r#else {
                    format!(
                        "if {} {} else {}",
                        cond.reparse(),
                        then.reparse(),
                        r#else.reparse()
                    )
                } else {
                    format!("if {} {}", cond.reparse(), then.reparse())
                }
            }
            Statement::While(cond, code) => {
                format!("while {} {}", cond.reparse(), code.reparse())
            }
            Statement::For(counter, iterator, code) => {
                format!("for {counter} in {} {}", iterator.reparse(), code.reparse())
            }
            Statement::Match(expr, cond) => {
                format!("match {} {{ {} }}", expr.reparse(), {
                    cond.iter()
                        .map(|case| {
                            format!(
                                "{} => {}",
                                case.0
                                    .iter()
                                    .map(|i| i.reparse())
                                    .collect::<Vec<String>>()
                                    .join(" | "),
                                case.1.reparse()
                            )
                        })
                        .collect::<Vec<String>>()
                        .join(", ")
                })
            }
            Statement::Fault => "fault".to_string(),
            Statement::Let(name, val) => format!("let {name} = {}", val.reparse()),
            Statement::Print(exprs) => format!(
                "print {}",
                exprs
                    .iter()
                    .map(|i| i.reparse())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}

#[derive(Debug, Clone)]
enum Expr {
    Infix(Box<Infix>),
    List(Vec<Expr>),
    Block(Program),
    Value(Type),
}

impl Expr {
    fn eval(&self, engine: &mut Engine) -> Option<Type> {
        Some(match self {
            Expr::Infix(infix) => (*infix).eval(engine)?,
            Expr::Block(block) => engine.eval(block.clone())?,
            Expr::List(list) => {
                let mut result = vec![];
                for i in list {
                    result.push(i.eval(engine)?)
                }
                Type::List(result)
            }
            Expr::Value(value) => {
                if let Type::Symbol(name) = value {
                    if let Some(refer) = engine.scope.get(name.as_str()) {
                        refer.clone()
                    } else {
                        value.clone()
                    }
                } else {
                    value.clone()
                }
            }
        })
    }

    fn parse(soruce: String) -> Option<Expr> {
        let token_list: Vec<String> = tokenize(soruce, SPACE.to_vec())?;
        let token = token_list.last()?.trim().to_string();
        let token = if let Ok(n) = token.parse::<f64>() {
            Expr::Value(Type::Number(n))
        } else if token.starts_with('(') && token.ends_with(')') {
            let token = token.get(1..token.len() - 1)?.to_string();
            Expr::parse(token)?
        } else if token.starts_with('{') && token.ends_with('}') {
            let token = token.get(1..token.len() - 1)?.to_string();
            Expr::Block(Engine::parse(token)?)
        } else if token.starts_with('[') && token.ends_with(']') {
            let token = token.get(1..token.len() - 1)?.to_string();
            let mut list = vec![];
            for elm in tokenize(token, vec![','])? {
                list.push(Expr::parse(elm.trim().to_string())?);
            }
            Expr::List(list)
        } else if token.starts_with('"') && token.ends_with('"') {
            let token = token.get(1..token.len() - 1)?.to_string();
            Expr::Value(Type::Text(token))
        } else if token.starts_with('λ') && token.contains('.') {
            let token = token.replacen("λ", "", 1);
            let (arg, body) = token.split_once(".")?;
            Expr::Value(Type::Function(Function::UserDefined(
                arg.to_string(),
                Box::new(Expr::parse(body.to_string())?),
            )))
        } else if token.starts_with('\\') && token.contains('.') {
            let token = token.replacen('\\', "", 1);
            let (arg, body) = token.split_once(".")?;
            Expr::Value(Type::Function(Function::UserDefined(
                arg.to_string(),
                Box::new(Expr::parse(body.to_string())?),
            )))
        } else if token.contains('(') && token.ends_with(')') {
            let token = token.get(..token.len() - 1)?.to_string();
            let (name, arg) = token.split_once("(")?;
            Expr::Infix(Box::new(Infix {
                operator: Operator::Apply,
                values: (
                    Expr::parse(name.to_string())?,
                    Expr::parse(arg.to_string())?,
                ),
            }))
        } else {
            Expr::Value(Type::Symbol(token))
        };

        if let Some(operator) = token_list
            .len()
            .checked_sub(2)
            .and_then(|idx| token_list.get(idx))
        {
            let is_operator: fn(String) -> Option<Operator> = |operator| {
                Some(match operator.as_str() {
                    "+" => Operator::Add,
                    "-" => Operator::Sub,
                    "*" => Operator::Mul,
                    "/" => Operator::Div,
                    "%" => Operator::Mod,
                    "^" => Operator::Pow,
                    "==" => Operator::Equal,
                    "!=" => Operator::NotEq,
                    "<" => Operator::LessThan,
                    "<=" => Operator::LessThanEq,
                    ">" => Operator::GreaterThan,
                    ">=" => Operator::GreaterThanEq,
                    "&" => Operator::And,
                    "|" => Operator::Or,
                    "::" => Operator::Access,
                    "as" => Operator::As,
                    "@" => Operator::Apply,
                    _ => return None,
                })
            };
            if let Some(operator) = is_operator(operator.to_string()) {
                let mut result = Some(Expr::Infix(Box::new(Infix {
                    operator,
                    values: (
                        Expr::parse(token_list.get(..token_list.len() - 2)?.join(" "))?,
                        token,
                    ),
                })))?;
                result.optimize();
                Some(result)
            } else {
                let mut result = Some(Expr::Infix(Box::new(Infix {
                    operator: Operator::Apply,
                    values: (
                        Expr::parse(token_list.get(..token_list.len() - 1)?.join(" "))?,
                        token,
                    ),
                })))?;
                result.optimize();
                Some(result)
            }
        } else {
            return Some(token);
        }
    }

    fn optimize(&mut self) {
        if let Expr::Infix(infix) = self {
            if let Infix {
                operator: Operator::Add,
                values: (expr, Expr::Value(Type::Number(0.0))),
            }
            | Infix {
                operator: Operator::Add,
                values: (Expr::Value(Type::Number(0.0)), expr),
            }
            | Infix {
                operator: Operator::Mul,
                values: (expr, Expr::Value(Type::Number(1.0))),
            }
            | Infix {
                operator: Operator::Mul,
                values: (Expr::Value(Type::Number(1.0)), expr),
            }
            | Infix {
                operator: Operator::Sub,
                values: (expr, Expr::Value(Type::Number(0.0))),
            } = *infix.clone()
            {
                *self = expr.clone();
            } else if let Infix {
                operator: Operator::Access,
                values: (Expr::List(list), mut index),
            } = *infix.clone()
            {
                index.optimize();
                if let Expr::Value(Type::Number(index)) = index {
                    if let Some(expr) = list.get(index as usize) {
                        let mut expr = expr.clone();
                        expr.optimize();
                        *self = expr.clone()
                    }
                }
            } else if let Infix {
                operator: Operator::Add,
                values: (Expr::Value(Type::Number(a)), Expr::Value(Type::Number(b))),
            } = *infix.clone()
            {
                *self = Expr::Value(Type::Number(a + b));
            } else if let Infix {
                operator: Operator::Sub,
                values: (Expr::Value(Type::Number(a)), Expr::Value(Type::Number(b))),
            } = *infix.clone()
            {
                *self = Expr::Value(Type::Number(a - b));
            } else if let Infix {
                operator: Operator::Mul,
                values: (Expr::Value(Type::Number(a)), Expr::Value(Type::Number(b))),
            } = *infix.clone()
            {
                *self = Expr::Value(Type::Number(a * b));
            } else if let Infix {
                operator: Operator::Div,
                values: (Expr::Value(Type::Number(a)), Expr::Value(Type::Number(b))),
            } = *infix.clone()
            {
                *self = Expr::Value(Type::Number(a / b));
            }
        } else if let Expr::List(exprs) = self {
            for expr in exprs {
                expr.optimize();
            }
        }
    }

    fn reparse(&self) -> String {
        match self {
            Expr::List(list) => format!(
                "[{}]",
                list.iter()
                    .map(|i| i.reparse())
                    .collect::<Vec<String>>()
                    .join(", "),
            ),
            Expr::Infix(infix) => format!("({})", infix.reparse()),
            Expr::Value(val) => val.get_symbol(),
            Expr::Block(block) => format!(
                "{{ {} }}",
                block
                    .iter()
                    .map(|i| i.reparse())
                    .collect::<Vec<String>>()
                    .join("; ")
            ),
        }
    }
}

#[derive(Clone, Debug)]
enum Function {
    BuiltIn(fn(Type, &mut Engine) -> Option<Type>),
    UserDefined(String, Box<Expr>),
}

#[derive(Debug, Clone)]
struct Infix {
    operator: Operator,
    values: (Expr, Expr),
}

impl Infix {
    fn eval(&self, engine: &mut Engine) -> Option<Type> {
        let left = self.values.0.eval(engine);
        let right = self.values.1.eval(engine);

        Some(match self.operator {
            Operator::Add => {
                if let (Some(Type::Number(left)), Some(Type::Number(right))) = (&left, &right) {
                    Type::Number(left + right)
                } else if let (Some(Type::Text(left)), Some(Type::Text(right))) = (&left, &right) {
                    Type::Text(left.clone() + right)
                } else if let (Some(Type::List(mut left)), Some(right)) = (left, right) {
                    left.push(right);
                    Type::List(left)
                } else {
                    return None;
                }
            }
            Operator::Sub => {
                if let (Some(Type::Number(left)), Some(Type::Number(right))) = (&left, &right) {
                    Type::Number(left - right)
                } else if let (Some(Type::Text(left)), Some(Type::Text(right))) = (&left, &right) {
                    Type::Text(left.replace(right, ""))
                } else if let (Some(Type::List(mut left)), Some(Type::Number(right))) =
                    (left, right)
                {
                    left.remove(right as usize);
                    Type::List(left)
                } else {
                    return None;
                }
            }
            Operator::Mul => {
                if let (Some(Type::Number(left)), Some(Type::Number(right))) = (&left, &right) {
                    Type::Number(left * right)
                } else if let (Some(Type::Text(left)), Some(Type::Number(right))) = (left, right) {
                    Type::Text(left.repeat(right as usize))
                } else {
                    return None;
                }
            }
            Operator::Div => Type::Number(left?.get_number()? / right?.get_number()?),
            Operator::Mod => Type::Number(left?.get_number()? % right?.get_number()?),
            Operator::Pow => Type::Number(left?.get_number()?.powf(right?.get_number()?)),
            Operator::Equal => {
                if left?.get_symbol() == right.clone()?.get_symbol() {
                    right?
                } else {
                    return None;
                }
            }
            Operator::NotEq => {
                if left?.get_symbol() != right.clone()?.get_symbol() {
                    right?
                } else {
                    return None;
                }
            }
            Operator::LessThan => {
                if left.clone()?.get_number() < right.clone()?.get_number() {
                    right?
                } else {
                    return None;
                }
            }
            Operator::LessThanEq => {
                if left?.get_number() <= right.clone()?.get_number() {
                    right?
                } else {
                    return None;
                }
            }
            Operator::GreaterThan => {
                if left?.get_number() > right.clone()?.get_number() {
                    right?
                } else {
                    return None;
                }
            }
            Operator::GreaterThanEq => {
                if left?.get_number() >= right.clone()?.get_number() {
                    right?
                } else {
                    return None;
                }
            }
            Operator::And => {
                if left.is_some() && right.is_some() {
                    right?
                } else {
                    return None;
                }
            }
            Operator::Or => {
                if left.is_some() || right.is_some() {
                    right.unwrap_or(left?)
                } else {
                    return None;
                }
            }
            Operator::Access => {
                if let Some(Type::List(list)) = left.clone() {
                    let index = right?.get_number()?;
                    list.get(index as usize)?.clone()
                } else if let Some(Type::Text(text)) = left {
                    let index = right?.get_number()?;
                    Type::Text(
                        text.chars()
                            .collect::<Vec<char>>()
                            .get(index as usize)?
                            .to_string(),
                    )
                } else {
                    return None;
                }
            }
            Operator::As => match right?.get_text().as_str() {
                "number" => Type::Number(left?.get_number()?),
                "symbol" => Type::Symbol(left?.get_symbol()),
                "text" => Type::Text(left?.get_text()),
                "list" => Type::List(left?.get_list()),
                _ => return None,
            },
            Operator::Apply => match left?.get_function()? {
                Function::BuiltIn(func) => func(right?, engine)?,
                Function::UserDefined(paramater, code) => {
                    engine.scope.insert(paramater.to_string(), right?);
                    code.eval(engine)?
                }
            },
        })
    }

    fn reparse(&self) -> String {
        let operator = match self.operator {
            Operator::Add => "+",
            Operator::Sub => "-",
            Operator::Mul => "*",
            Operator::Div => "/",
            Operator::Mod => "%",
            Operator::Pow => "^",
            Operator::Equal => "==",
            Operator::NotEq => "!=",
            Operator::LessThan => "<",
            Operator::LessThanEq => "<=",
            Operator::GreaterThan => ">",
            Operator::GreaterThanEq => ">=",
            Operator::And => "&",
            Operator::Or => "|",
            Operator::Access => "::",
            Operator::As => "as",
            Operator::Apply => "",
        }
        .to_string();
        if let Expr::Infix(infix) = self.values.1.clone() {
            format!(
                "{} {operator} ({})",
                self.values.0.reparse(),
                infix.reparse()
            )
        } else {
            format!(
                "{} {operator} {}",
                self.values.0.reparse(),
                self.values.1.reparse()
            )
        }
    }
}

#[derive(Debug, Clone)]
enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Equal,
    NotEq,
    LessThan,
    LessThanEq,
    GreaterThan,
    GreaterThanEq,
    And,
    Or,
    Access,
    As,
    Apply,
}

#[derive(Debug, Clone)]
enum Type {
    Number(f64),
    Symbol(String),
    Text(String),
    List(Vec<Type>),
    Function(Function),
    Null,
}

impl Type {
    fn get_number(&self) -> Option<f64> {
        match self {
            Type::Number(n) => Some(n.to_owned()),
            Type::Symbol(s) | Type::Text(s) => {
                if let Ok(n) = s.trim().parse::<f64>() {
                    Some(n)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn get_symbol(&self) -> String {
        match self {
            Type::Symbol(s) => s.to_string(),
            Type::Text(s) => format!("\"{s}\""),
            Type::Number(n) => n.to_string(),
            Type::Null => "null".to_string(),
            Type::Function(Function::BuiltIn(obj)) => format!("λx.{obj:?}"),
            Type::Function(Function::UserDefined(arg, code)) => {
                format!("λ{arg}.{}", code.reparse())
            }
            Type::List(l) => format!(
                "[{}]",
                l.iter()
                    .map(|i| i.get_symbol())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    fn get_text(&self) -> String {
        match self {
            Type::Number(n) => n.to_string(),
            Type::Symbol(s) | Type::Text(s) => s.to_string(),
            _ => String::new(),
        }
    }

    fn get_list(&self) -> Vec<Type> {
        match self {
            Type::List(list) => list.to_owned(),
            Type::Text(text) => text.chars().map(|i| Type::Text(i.to_string())).collect(),
            other => vec![other.to_owned()],
        }
    }

    fn get_function(&self) -> Option<Function> {
        if let Type::Function(func) = self {
            Some(func.to_owned())
        } else {
            None
        }
    }

    fn is_match(&self, condition: &Type) -> bool {
        if let (Type::List(list), Type::List(conds)) = (self, condition) {
            for (elm, cond) in list.iter().zip(conds) {
                if !elm.is_match(cond) {
                    return false;
                }
            }
            true
        } else {
            if condition.get_symbol() == "_" {
                return true;
            } else {
                self.get_symbol() == condition.get_symbol()
            }
        }
    }
}

fn tokenize(input: String, delimiter: Vec<char>) -> Option<Vec<String>> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current_token = String::new();
    let mut in_parentheses: usize = 0;
    let mut in_quote = false;

    for c in input.chars() {
        match c {
            '(' | '{' | '[' if !in_quote => {
                current_token.push(c);
                in_parentheses += 1;
            }
            ')' | '}' | ']' if !in_quote => {
                current_token.push(c);
                if in_parentheses > 0 {
                    in_parentheses -= 1;
                } else {
                    return None;
                }
            }
            '"' => {
                in_quote = !in_quote;
                current_token.push(c);
            }
            other => {
                if delimiter.contains(&other) && !in_quote {
                    if in_parentheses != 0 {
                        current_token.push(c);
                    } else if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                } else {
                    current_token.push(c);
                }
            }
        }
    }

    // Syntax error check
    if in_quote || in_parentheses != 0 {
        return None;
    }

    if in_parentheses == 0 && !current_token.is_empty() {
        tokens.push(current_token.clone());
        current_token.clear();
    }

    Some(tokens)
}
