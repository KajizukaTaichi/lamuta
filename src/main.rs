use colored::*;
use std::{
    collections::HashMap,
    env::{current_dir, set_current_dir},
    fs::{create_dir_all, read_dir, read_to_string, File},
    io::{self, Write},
    path::Path,
    process::exit,
};

const VERSION: &str = "0.2.0";
const SPACE: [char; 5] = [' ', '　', '\n', '\t', '\r'];

fn main() {
    println!("{title} {VERSION}", title = "Lamuta".blue().bold());
    let (mut engine, mut code) = (Engine::new(), String::new());
    let (mut session, mut line) = (1, 1);

    loop {
        print!("[{session}:{line}]> ");
        io::stdout().flush().unwrap();

        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
        let buffer = buffer.trim().to_string();

        if buffer == ":q" {
            code.clear();
            session += 1;
            line = 1;
            continue;
        } else if buffer == ":env" {
            println!("Defined variables:");
            let width = &engine
                .env
                .keys()
                .map(|i| i.chars().count())
                .max()
                .unwrap_or(0);
            for (k, v) in &engine.env {
                if let Type::Function(_, Function::BuiltIn(_)) = v {
                } else {
                    println!(" {k:<width$} = {}", v.get_symbol());
                }
            }
            continue;
        }

        code += &format!("{buffer}\n");
        if let Some(ast) = Engine::parse(code.clone()) {
            if let Some(result) = engine.eval(ast) {
                println!("{navi} {}", result.get_symbol(), navi = "=>".green());
            } else {
                println!("{navi} {}", "Fault", navi = "=>".red());
            }
            code.clear();
            session += 1;
            line = 1;
        } else {
            line += 1;
        }
    }
}

type Scope = HashMap<String, Type>;
type Program = Vec<Statement>;
#[derive(Debug, Clone)]
struct Engine {
    env: Scope,
    project: Option<String>,
}

impl Engine {
    fn new() -> Engine {
        Engine {
            project: None,
            env: HashMap::from([
                (
                    "type".to_string(),
                    Type::Function(
                        None,
                        Function::BuiltIn(|expr, _| Some(Type::Signature(expr.get_type()?))),
                    ),
                ),
                (
                    "save".to_string(),
                    Type::Function(
                        None,
                        Function::BuiltIn(|arg, engine| {
                            let mut render = String::new();
                            for (k, v) in &engine.env {
                                if let Type::Function(None, Function::BuiltIn(_)) = v {
                                } else {
                                    render += &format!("let {k} = {};\n", v.get_symbol());
                                }
                            }
                            if let Ok(mut file) = File::create(arg.get_text()) {
                                if file.write_all(render.as_bytes()).is_ok() {
                                    Some(Type::Text("Saved environment".to_string()))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }),
                    ),
                ),
                (
                    "newProject".to_string(),
                    Type::Function(
                        None,
                        Function::BuiltIn(|arg, engine| {
                            let name = arg.get_text();
                            let path = Path::new(&name);
                            let home = current_dir().unwrap_or_default();
                            if create_dir_all(path).is_ok() {
                                if set_current_dir(path).is_ok() {
                                    if create_dir_all("lib").is_err() {
                                        return None;
                                    }
                                    if create_dir_all("src").is_ok() {
                                        if let Ok(mut file) =
                                            File::create_new(Path::new("src/main.lm"))
                                        {
                                            if file
                                                .write_all(r#"print "Hello, world!""#.as_bytes())
                                                .is_ok()
                                            {
                                                engine.project = Some(name.clone());
                                                set_current_dir(home).unwrap_or_default();
                                                Some(Type::Text(format!("Created project: {name}")))
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }),
                    ),
                ),
                (
                    "login".to_string(),
                    Type::Function(
                        None,
                        Function::BuiltIn(|arg, engine| {
                            let path = arg.get_text();
                            if Path::new(&path).exists() {
                                engine.project = Some(path.clone());
                                Some(Type::Text(format!("Logged in project: {path}")))
                            } else {
                                None
                            }
                        }),
                    ),
                ),
                (
                    "logout".to_string(),
                    Type::Function(
                        None,
                        Function::BuiltIn(|_, engine| {
                            engine.project = None;
                            Some(Type::Text("Logged out current project".to_string()))
                        }),
                    ),
                ),
                (
                    "run".to_string(),
                    Type::Function(
                        None,
                        Function::BuiltIn(|_, engine| {
                            if let Some(project_path) = engine.project.clone() {
                                let home = current_dir().unwrap_or_default();
                                if set_current_dir(project_path).is_ok() {
                                    if let Ok(dir) = read_dir(Path::new("lib")) {
                                        for entry in dir {
                                            if let Ok(entry) = entry {
                                                let lib_file = entry.file_name();
                                                if let Ok(code) = read_to_string(Path::new(
                                                    &format!("lib/{}", lib_file.to_str()?),
                                                )) {
                                                    engine.eval(Engine::parse(code)?);
                                                }
                                            }
                                        }
                                    }
                                    if let Ok(code) = read_to_string(Path::new("src/main.lm")) {
                                        set_current_dir(home).unwrap_or_default();
                                        engine.eval(Engine::parse(code)?)
                                    } else {
                                        set_current_dir(home).unwrap_or_default();
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }),
                    ),
                ),
                (
                    "load".to_string(),
                    Type::Function(
                        None,
                        Function::BuiltIn(|expr, engine| {
                            let path = expr.get_text();
                            let path = Path::new(&path);
                            if let Ok(module) = read_to_string(path) {
                                let home = current_dir().unwrap_or_default();
                                if let Some(parent_dir) = path.parent() {
                                    set_current_dir(parent_dir).unwrap_or_default();
                                }
                                let module = Engine::parse(module)?;
                                let result = engine.eval(module)?;
                                set_current_dir(home).unwrap_or_default();
                                Some(result)
                            } else {
                                None
                            }
                        }),
                    ),
                ),
                (
                    "input".to_string(),
                    Type::Function(
                        None,
                        Function::BuiltIn(|expr, _| {
                            let prompt = expr.get_text();
                            print!("{prompt}");
                            io::stdout().flush().unwrap();
                            let mut buffer = String::new();
                            if io::stdin().read_line(&mut buffer).is_ok() {
                                Some(Type::Text(buffer.trim().to_string()))
                            } else {
                                None
                            }
                        }),
                    ),
                ),
                (
                    "range".to_string(),
                    Type::Function(
                        None,
                        Function::BuiltIn(|params, _| {
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
                        }),
                    ),
                ),
                (
                    "exit".to_string(),
                    Type::Function(
                        None,
                        Function::BuiltIn(|arg, _| exit(arg.get_number()? as i32)),
                    ),
                ),
                ("doubleQuote".to_string(), Type::Text("\"".to_string())),
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
                    println!();
                    io::stdout().flush().unwrap();
                    Type::Null
                }
                Statement::Let(name, expr) => {
                    let val = expr.eval(self)?;
                    if name != "_" {
                        self.env.insert(name, val.clone());
                    }
                    val
                }
                Statement::If(expr, then, r#else) => {
                    if let Some(it) = expr.eval(self) {
                        self.env.insert("it".to_string(), it);
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
                        self.env.insert("it".to_string(), it);
                        result = code.eval(self)?;
                    }
                    result
                }
                Statement::For(counter, expr, code) => {
                    let mut result = Type::Null;
                    for i in expr.eval(self)?.get_list() {
                        if counter != "_" {
                            self.env.insert(counter.clone(), i);
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
                let tokens = tokenize(i, SPACE.to_vec())?;
                let pos = tokens.iter().position(|i| i == "=>")?;
                let mut cond = vec![];
                for i in tokenize(tokens.get(..pos)?.join(" "), vec!['|'])? {
                    cond.push(Expr::parse(i.to_string())?)
                }
                conds.push((cond, Expr::parse(tokens.get(pos + 1..)?.join(" "))?))
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

    fn format(&self) -> String {
        match self {
            Statement::Value(expr) => format!("{}", expr.format()),
            Statement::If(cond, then, r#else) => {
                if let Some(r#else) = r#else {
                    format!(
                        "if {} {} else {}",
                        cond.format(),
                        then.format(),
                        r#else.format()
                    )
                } else {
                    format!("if {} {}", cond.format(), then.format())
                }
            }
            Statement::While(cond, code) => {
                format!("while {} {}", cond.format(), code.format())
            }
            Statement::For(counter, iterator, code) => {
                format!("for {counter} in {} {}", iterator.format(), code.format())
            }
            Statement::Match(expr, cond) => {
                format!("match {} {{ {} }}", expr.format(), {
                    cond.iter()
                        .map(|case| {
                            format!(
                                "{} => {}",
                                case.0
                                    .iter()
                                    .map(|i| i.format())
                                    .collect::<Vec<String>>()
                                    .join(" | "),
                                case.1.format()
                            )
                        })
                        .collect::<Vec<String>>()
                        .join(", ")
                })
            }
            Statement::Fault => "fault".to_string(),
            Statement::Let(name, val) => {
                format!("let {name} = {}", val.format())
            }
            Statement::Print(exprs) => format!(
                "print {}",
                exprs
                    .iter()
                    .map(|i| i.format())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}

#[derive(Debug, Clone)]
enum Expr {
    Derefer(Box<Expr>),
    Infix(Box<Infix>),
    List(Vec<Expr>),
    Struct(Vec<(Expr, Expr)>),
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
            Expr::Struct(st) => {
                let mut result = HashMap::new();
                for (k, x) in st {
                    result.insert(k.eval(engine)?.get_text(), x.eval(engine)?);
                }
                Type::Struct(None, result)
            }
            Expr::Value(Type::Signature(Signature::Function(arg, sig))) => Type::Signature(
                Signature::Function(Box::new(*arg.clone()), Box::new(*sig.clone())),
            ),
            Expr::Value(Type::Signature(sig)) => Type::Signature(sig.clone()),
            Expr::Value(Type::Symbol(name)) => {
                if let Some(refer) = engine.env.get(name.as_str()) {
                    refer.clone()
                } else {
                    Type::Symbol(name.to_owned())
                }
            }
            Expr::Value(value) => value.clone(),
            Expr::Derefer(pointer) => match pointer.clone().eval(engine)? {
                Type::Refer(to) => Expr::Value(Type::Symbol(to.to_string())).eval(engine)?,
                _ => return None,
            },
        })
    }

    fn parse(source: String) -> Option<Expr> {
        let token_list: Vec<String> = tokenize(source, SPACE.to_vec())?;
        let token = token_list.last()?.trim().to_string();
        let token = if let Ok(n) = token.parse::<f64>() {
            Expr::Value(Type::Number(n))
        } else if let Some(sig) = Signature::parse(token.clone()) {
            Expr::Value(Type::Signature(sig))
        } else if token.starts_with('(') && token.ends_with(')') {
            let token = token.get(1..token.len() - 1)?.to_string();
            Expr::parse(token)?
        } else if token.starts_with('{') && token.ends_with('}') {
            let token = token.get(1..token.len() - 1)?.to_string();
            let parse_struct = || {
                let mut result = Vec::new();
                for i in tokenize(token.clone(), vec![','])? {
                    let splited = tokenize(i, vec![':'])?;
                    result.push((
                        Expr::parse(splited.get(0)?.to_string())?,
                        Expr::parse(splited.get(1)?.to_string())?,
                    ));
                }
                Some(Expr::Struct(result))
            };
            if let Some(st) = parse_struct() {
                st
            } else {
                Expr::Block(Engine::parse(token)?)
            }
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
            Expr::Value(Type::Function(
                None,
                Function::UserDefined(arg.to_string(), Box::new(Expr::parse(body.to_string())?)),
            ))
        } else if token.starts_with('\\') && token.contains('.') {
            let token = token.replacen('\\', "", 1);
            let (arg, body) = token.split_once(".")?;
            Expr::Value(Type::Function(
                None,
                Function::UserDefined(arg.to_string(), Box::new(Expr::parse(body.to_string())?)),
            ))
        } else if token.starts_with("fn(") && token.contains("->") && token.ends_with(")") {
            let token = token.replacen("fn(", "", 1);
            let token = token.get(..token.len() - 1)?.to_string();
            let (arg, body) = token.split_once("->")?;
            Expr::Value(Type::Function(
                None,
                Function::UserDefined(
                    arg.trim().to_string(),
                    Box::new(Expr::parse(body.to_string())?),
                ),
            ))
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
        } else if token.starts_with("&") {
            let token = token.replacen("&", "", 1);
            Expr::Value(Type::Refer(token))
        } else if token.starts_with("*") {
            let token = token.replacen("*", "", 1);
            Expr::Derefer(Box::new(Expr::parse(token)?))
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
                    "=" => Operator::Assign,
                    "bind" | ":" => Operator::Bind,
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
            } else if operator.starts_with("`") && operator.ends_with("`") {
                let operator = operator[1..operator.len() - 1].to_string();
                let mut result = Some(Expr::Infix(Box::new(Infix {
                    operator: Operator::Apply,
                    values: (
                        Expr::Infix(Box::new(Infix {
                            operator: Operator::Apply,
                            values: (
                                Expr::parse(operator)?,
                                Expr::parse(token_list.get(..token_list.len() - 2)?.join(" "))?,
                            ),
                        })),
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
            Some(token)
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

    fn format(&self) -> String {
        match self {
            Expr::List(list) => format!(
                "[{}]",
                list.iter()
                    .map(|i| i.format())
                    .collect::<Vec<String>>()
                    .join(", "),
            ),
            Expr::Infix(infix) => format!("({})", infix.format()),
            Expr::Value(val) => val.get_symbol(),
            Expr::Block(block) => format!(
                "{{ {} }}",
                block
                    .iter()
                    .map(|i| i.format())
                    .collect::<Vec<String>>()
                    .join("; ")
            ),
            Expr::Struct(st) => format!(
                "{{ {} }}",
                st.iter()
                    .map(|(k, x)| format!("{}: {}", k.format(), x.format()))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Expr::Derefer(to) => format!("*{}", to.format()),
        }
    }

    fn replace(&self, from: &Expr, to: &Expr) -> Expr {
        match self {
            Expr::List(list) => Expr::List(
                list.iter()
                    .map(|i| i.replace(from, to))
                    .collect::<Vec<Expr>>(),
            ),
            Expr::Struct(st) => Expr::Struct(
                st.iter()
                    .map(|(k, x)| (k.replace(from, to), x.replace(from, to)))
                    .collect::<Vec<(Expr, Expr)>>(),
            ),
            Expr::Infix(infix) => Expr::Infix(Box::new(Infix {
                operator: infix.operator.clone(),
                values: (
                    infix.values.0.replace(from, to),
                    infix.values.1.replace(from, to),
                ),
            })),
            Expr::Block(block) => Expr::Block(
                block
                    .iter()
                    .map(|i| match i {
                        Statement::If(cond, then, r#else) => Statement::If(
                            cond.replace(from, to),
                            then.replace(from, to),
                            r#else.clone().and_then(|j| Some(j.replace(from, to))),
                        ),
                        Statement::While(cond, code) => {
                            Statement::While(cond.replace(from, to), code.replace(from, to))
                        }
                        Statement::Value(val) => Statement::Value(val.replace(from, to)),
                        Statement::Match(expr, cond) => Statement::Match(
                            expr.replace(from, to),
                            cond.iter()
                                .map(|j| {
                                    (
                                        j.0.iter().map(|k| k.replace(from, to)).collect(),
                                        j.1.replace(from, to),
                                    )
                                })
                                .collect(),
                        ),
                        Statement::Print(vals) => {
                            Statement::Print(vals.iter().map(|j| j.replace(from, to)).collect())
                        }
                        Statement::For(counter, iter, code) => Statement::For(
                            counter.clone(),
                            iter.replace(from, to),
                            code.replace(from, to),
                        ),
                        Statement::Let(name, val) => {
                            Statement::Let(name.clone(), val.replace(from, to))
                        }
                        Statement::Fault => Statement::Fault,
                    })
                    .collect(),
            ),
            Expr::Value(Type::Function(None, Function::UserDefined(arg, func))) => {
                Expr::Value(Type::Function(
                    None,
                    Function::UserDefined(
                        arg.to_string(),
                        if from.format() != *arg {
                            Box::new(func.replace(from, to))
                        } else {
                            func.clone()
                        },
                    ),
                ))
            }
            Expr::Value(val) => {
                if let (Type::Symbol(val), Expr::Value(Type::Symbol(from))) = (val, from) {
                    if val == from {
                        to.clone()
                    } else {
                        self.clone()
                    }
                } else {
                    self.clone()
                }
            }
            Expr::Derefer(to) => Expr::Derefer(Box::new(to.replace(from, to))),
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
                    Type::Text(left.clone() + &right)
                } else if let (Some(Type::List(left)), Some(Type::List(right))) = (&left, &right) {
                    Type::List([left.clone(), right.clone()].concat())
                } else if let (
                    Some(Type::Struct(None, mut left)),
                    Some(Type::Struct(None, right)),
                ) = (left.clone(), &right)
                {
                    left.extend(right.clone());
                    Type::Struct(None, left)
                } else {
                    return None;
                }
            }
            Operator::Sub => {
                if let (Some(Type::Number(left)), Some(Type::Number(right))) = (&left, &right) {
                    Type::Number(left - right)
                } else if let (Some(Type::Text(left)), Some(Type::Text(right))) = (&left, &right) {
                    Type::Text(left.replacen(right, "", 1))
                } else if let (Some(Type::List(mut left)), Some(Type::List(right))) =
                    (left.clone(), &right)
                {
                    let first_index = left.windows(right.len()).position(|i| {
                        i.iter().map(|j| j.get_symbol()).collect::<Vec<_>>()
                            == right.iter().map(|j| j.get_symbol()).collect::<Vec<_>>()
                    })?;
                    for _ in 0..right.len() {
                        left.remove(first_index);
                    }
                    Type::List(left)
                } else if let (Some(Type::List(mut left)), Some(Type::Number(right))) =
                    (left.clone(), &right)
                {
                    left.remove(right.clone() as usize);
                    Type::List(left)
                } else if let (Some(Type::Text(mut left)), Some(Type::Number(right))) =
                    (left, right)
                {
                    left.remove(right as usize);
                    Type::Text(left)
                } else {
                    return None;
                }
            }
            Operator::Mul => {
                if let (Some(Type::Number(left)), Some(Type::Number(right))) = (&left, &right) {
                    Type::Number(left * right)
                } else if let (Some(Type::Text(left)), Some(Type::Number(right))) = (&left, &right)
                {
                    Type::Text(left.repeat(*right as usize))
                } else if let (Some(Type::List(left)), Some(Type::Number(right))) = (left, right) {
                    Type::List((0..right as usize).flat_map(|_| left.clone()).collect())
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
                if let (Some(Type::List(list)), Some(Type::Number(index))) =
                    (left.clone(), right.clone())
                {
                    list.get(index as usize)?.clone()
                } else if let (Some(Type::Text(text)), Some(Type::Number(index))) =
                    (left.clone(), right.clone())
                {
                    Type::Text(
                        text.chars()
                            .collect::<Vec<char>>()
                            .get(index as usize)?
                            .to_string(),
                    )
                } else if let (Some(Type::Struct(_, st)), Some(Type::Text(index))) =
                    (left.clone(), right.clone())
                {
                    st.get(&index)?.clone()
                } else {
                    return None;
                }
            }
            Operator::As => match right?.get_signature()? {
                Signature::Number => Type::Number(left?.get_number()?),
                Signature::Symbol => Type::Symbol(left?.get_symbol()),
                Signature::Text => Type::Text(left?.get_text()),
                Signature::List => Type::List(left?.get_list()),
                Signature::Function(arg, sig) => {
                    Type::Function(Some(Signature::Function(arg, sig)), left?.get_function()?.1)
                }
                _ => return None,
            },
            Operator::Apply => match left?.get_function()? {
                (Some(sig), Function::BuiltIn(func)) => {
                    if let Signature::Function(arg, ret) = sig {
                        let result = func(
                            if right.clone()?.get_type()?.format() == arg.format() {
                                right?
                            } else {
                                return None;
                            },
                            engine,
                        )?;
                        if result.get_type()?.format() == ret.format() {
                            result
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                (Some(sig), Function::UserDefined(parameter, code)) => {
                    if let Signature::Function(arg, ret) = sig {
                        let code = code.replace(
                            &Expr::Value(Type::Symbol(parameter)),
                            &Expr::Value(if right.clone()?.get_type()?.format() == arg.format() {
                                right?
                            } else {
                                return None;
                            }),
                        );
                        let result = code.eval(&mut engine.clone())?;
                        if result.get_type()?.format() == ret.format() {
                            result
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                (None, Function::BuiltIn(func)) => func(right?, engine)?,
                (None, Function::UserDefined(parameter, code)) => {
                    let code =
                        code.replace(&Expr::Value(Type::Symbol(parameter)), &Expr::Value(right?));
                    code.eval(&mut engine.clone())?
                }
            },
            Operator::Bind => {
                let value = left?;
                let r#type = right?.get_signature()?;
                match r#type.clone() {
                    Signature::Function(arg, ret) => {
                        Type::Function(Some(Signature::Function(arg, ret)), value.get_function()?.1)
                    }
                    Signature::Struct(elms) => {
                        let st = value.get_struct()?.1;
                        if st.keys().cloned().collect::<Vec<String>>() == elms {
                            Type::Struct(Some(r#type), st.clone())
                        } else {
                            return None;
                        }
                    }
                    _ => {
                        if value.get_type()?.format() == r#type.format() {
                            value
                        } else {
                            return None;
                        }
                    }
                }
            }
            Operator::Assign => {
                let name = left?.get_text();
                let val = right?;
                if name != "_" {
                    engine.env.insert(name, val.clone());
                }
                val
            }
        })
    }

    fn format(&self) -> String {
        let is_operator = |op| {
            Some(
                match op {
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
                    Operator::Bind => "bind",
                    Operator::Assign => "=",
                    Operator::Apply => return None,
                }
                .to_string(),
            )
        };
        if let Some(op) = is_operator(self.operator.clone()) {
            if let Expr::Infix(infix) = self.values.1.clone() {
                format!("{} {op} ({})", self.values.0.format(), infix.format())
            } else {
                format!(
                    "{} {op} {}",
                    if let Expr::Infix(infix) = self.values.0.clone() {
                        infix.format()
                    } else {
                        self.values.0.format()
                    },
                    if let Expr::Infix(infix) = self.values.1.clone() {
                        infix.format()
                    } else {
                        self.values.1.format()
                    },
                )
            }
        } else {
            if let Expr::Infix(infix) = self.values.1.clone() {
                format!("{} ({})", self.values.0.format(), infix.format())
            } else {
                format!(
                    "{} {}",
                    if let Expr::Infix(infix) = self.values.0.clone() {
                        infix.format()
                    } else {
                        self.values.0.format()
                    },
                    if let Expr::Infix(infix) = self.values.1.clone() {
                        infix.format()
                    } else {
                        self.values.1.format()
                    },
                )
            }
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
    Bind,
    Assign,
}

#[derive(Debug, Clone)]
enum Type {
    Number(f64),
    Symbol(String),
    Refer(String),
    Text(String),
    List(Vec<Type>),
    Function(Option<Signature>, Function),
    Signature(Signature),
    Struct(Option<Signature>, HashMap<String, Type>),
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
            Type::Function(Some(sig), Function::BuiltIn(obj)) => {
                format!("(λx.{obj:?} bind {})", sig.format())
            }
            Type::Function(Some(sig), Function::UserDefined(arg, code)) => {
                format!("(λ{arg}.{} bind {})", code.format(), sig.format())
            }
            Type::Function(None, Function::BuiltIn(obj)) => format!("λx.{obj:?}"),
            Type::Function(None, Function::UserDefined(arg, code)) => {
                format!("λ{arg}.{}", code.format())
            }
            Type::List(l) => format!(
                "[{}]",
                l.iter()
                    .map(|i| i.get_symbol())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Type::Signature(sig) => sig.format(),
            Type::Struct(Some(sig), val) => format!(
                "({{ {1} }} bind {0})",
                sig.format(),
                val.iter()
                    .map(|(k, v)| format!("\"{k}\": {}", v.get_symbol()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Type::Refer(to) => {
                format!("&{to}")
            }
            Type::Struct(None, val) => format!(
                "{{ {} }}",
                val.iter()
                    .map(|(k, v)| format!("\"{k}\": {}", v.get_symbol()))
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

    fn get_function(&self) -> Option<(Option<Signature>, Function)> {
        if let Type::Function(sig, func) = self {
            Some((sig.clone(), func.to_owned()))
        } else {
            None
        }
    }

    fn get_type(&self) -> Option<Signature> {
        Some(match self {
            Type::Number(_) => Signature::Number,
            Type::Text(_) => Signature::Text,
            Type::Symbol(_) => Signature::Symbol,
            Type::List(_) => Signature::List,
            Type::Struct(Some(sig), _) => sig.clone(),
            Type::Function(Some(Signature::Function(arg_sig, ret_sig)), _) => {
                Signature::Function(arg_sig.to_owned(), ret_sig.to_owned())
            }
            Type::Signature(_) => Signature::Signature,
            _ => return None,
        })
    }

    fn get_signature(&self) -> Option<Signature> {
        if let Type::Signature(sig) = self {
            Some(sig.clone())
        } else {
            None
        }
    }

    fn get_struct(&self) -> Option<(Option<Signature>, HashMap<String, Type>)> {
        if let Type::Struct(sig, val) = self {
            Some((sig.clone(), val.clone()))
        } else {
            None
        }
    }

    fn is_match(&self, condition: &Type) -> bool {
        if let (Type::List(list), Type::List(conds)) = (self, condition) {
            if list.len() != conds.len() {
                return false;
            }
            for (elm, cond) in list.iter().zip(conds) {
                if !elm.is_match(cond) {
                    return false;
                }
            }
            true
        } else if let (Type::Struct(_, strct), Type::Struct(_, conds)) = (self, condition) {
            if strct.len() != conds.len() {
                return false;
            }
            for (elm, cond) in strct.iter().zip(conds) {
                if elm.0 != cond.0 || !elm.1.is_match(cond.1) {
                    return false;
                }
            }
            true
        } else {
            if condition.get_symbol() == "_" {
                true
            } else {
                self.get_symbol() == condition.get_symbol()
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Signature {
    Number,
    Symbol,
    Text,
    List,
    Signature,
    Function(Box<Signature>, Box<Signature>),
    Struct(Vec<String>),
}

impl Signature {
    fn parse(token: String) -> Option<Signature> {
        Some(if token == "number" {
            Signature::Number
        } else if token == "text" {
            Signature::Text
        } else if token == "list" {
            Signature::List
        } else if token == "symbol" {
            Signature::Symbol
        } else if token.starts_with("struct(") && token.ends_with(")") {
            let token = &token.replacen("struct(", "", 1);
            let token = token.get(..token.len() - 1)?.to_string();
            Signature::Struct(
                tokenize(token, vec![','])?
                    .iter()
                    .map(|i| i.trim().to_string())
                    .collect(),
            )
        } else if token.starts_with('λ') || token.starts_with("fn") {
            let token = &token.replacen(if token.starts_with("λ") { "λ" } else { "fn" }, "", 1);
            let (args, body) =
                token.split_once(if token.starts_with("λ") { "." } else { "->" })?;
            if let (
                Some(Expr::Value(Type::Signature(arg_sig))),
                Some(Expr::Value(Type::Signature(ret_sig))),
            ) = (Expr::parse(args.to_string()), Expr::parse(body.to_string()))
            {
                Signature::Function(Box::new(arg_sig), Box::new(ret_sig))
            } else {
                return None;
            }
        } else {
            return None;
        })
    }

    fn format(&self) -> String {
        match self {
            Signature::Struct(vals) => format!("struct({})", vals.join(", ")),
            Signature::Function(arg, vals) => {
                format!("fn({} -> {})", arg.format(), vals.format())
            }
            other => format!("{other:?}").to_lowercase(),
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
            '"' | '`' => {
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
