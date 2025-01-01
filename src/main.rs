use colored::*;
use reqwest::blocking;
use std::{
    collections::HashMap,
    env::{current_dir, set_current_dir},
    fs::{create_dir_all, read_dir, read_to_string, File},
    io::{self, Write},
    path::Path,
    process::exit,
};

const VERSION: &str = "0.3.2";
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
                if let Type::Function(Function::BuiltIn(_)) = v {
                } else {
                    println!(" {k:<width$} = {}", v.get_symbol());
                }
            }
            continue;
        }

        code += &format!("{buffer}\n");
        if let Some(ast) = Engine::parse(code.clone()) {
            if let Some(result) = engine.eval(ast) {
                println!(
                    "{navi} {result}",
                    result = result.get_symbol(),
                    navi = "=>".green()
                );
            } else {
                println!("{navi} Fault", navi = "=>".red());
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
    project: Option<(String, Type)>,
}

impl Engine {
    fn new() -> Engine {
        Engine {
            project: None,
            env: HashMap::from([
                (
                    "type".to_string(),
                    Type::Function(Function::BuiltIn(|expr, _| {
                        Some(Type::Signature(expr.get_type()))
                    })),
                ),
                (
                    "input".to_string(),
                    Type::Function(Function::BuiltIn(|expr, _| {
                        let prompt = expr.get_text()?;
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
                (
                    "exit".to_string(),
                    Type::Function(Function::BuiltIn(|arg, _| exit(arg.get_number()? as i32))),
                ),
                ("doubleQuote".to_string(), Type::Text("\"".to_string())),
                (
                    "load".to_string(),
                    Type::Function(Function::BuiltIn(|expr, engine| {
                        let path = expr.get_text()?;
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
                    })),
                ),
                (
                    "save".to_string(),
                    Type::Function(Function::BuiltIn(|arg, engine| {
                        let mut render = String::new();
                        for (k, v) in &engine.env {
                            if let Type::Function(Function::UserDefined(_, _)) = v {
                                render += &format!("let {k} = {};\n", v.get_symbol());
                            }
                        }
                        if let Ok(mut file) = File::create(arg.get_text()?) {
                            if file.write_all(render.as_bytes()).is_ok() {
                                Some(Type::Text("Saved environment".to_string()))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })),
                ),
                (
                    "newProject".to_string(),
                    Type::Function(Function::BuiltIn(|arg, _| {
                        let name = arg.get_text()?;
                        let path = Path::new(&name);
                        let home = current_dir().unwrap_or_default();
                        if create_dir_all(path).is_ok() {
                            if set_current_dir(path).is_ok() {
                                if create_dir_all("lib").is_err() {
                                    return None;
                                }
                                if let Ok(mut file) = File::create("config.lm") {
                                    if file.write_all(
                                        Type::Struct(HashMap::from([
                                        ("name".to_string(), Type::Text(name.clone())),
                                        (
                                            "depend".to_string(),
                                            Type::List(vec![Type::Text(
                                                "https://kajizukataichi.github.io/lamuta/lib/std.lm".to_string(),
                                            )]),
                                        ),
                                    ])).get_symbol().as_bytes()).is_err(){return None};
                                }
                                if create_dir_all("src").is_ok() {
                                    if let Ok(mut file) = File::create_new(Path::new("src/main.lm"))
                                    {
                                        if file
                                            .write_all(r#"print "Hello, world!""#.as_bytes())
                                            .is_ok()
                                        {
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
                    })),
                ),
                (
                    "install".to_string(),
                    Type::Function(Function::BuiltIn(|_, engine| {
                        if let Some((_, depend)) = engine.project.clone() {
                            for i in depend.get_struct()?.get("depend")?.get_list() {
                                let path = i.get_text()?.trim().to_string();
                                if let Ok(res) = blocking::get(path.clone()) {
                                    if let Ok(code) = res.text() {
                                        if let Ok(mut file) = File::create(&format!(
                                            "{}/lib/{}",
                                            engine.project.clone()?.0,
                                            Path::new(&path).file_name()?.to_str()?.to_string()
                                        )) {
                                            if file.write_all(code.as_bytes()).is_err() {
                                                return None;
                                            };
                                        } else {
                                            return None;
                                        };
                                    } else {
                                        return None;
                                    }
                                } else {
                                    return None;
                                }
                            }
                            Some(Type::Text("Dependencies library is installed!".to_string()))
                        } else {
                            None
                        }
                    })),
                ),
                (
                    "login".to_string(),
                    Type::Function(Function::BuiltIn(|arg, engine| {
                        let path = arg.get_text()?.trim().to_string();
                        if Path::new(&path).exists() {
                            engine.project = Some((
                                path.clone(),
                                engine.eval(Engine::parse(
                                    if let Ok(code) = read_to_string(format!("{path}/config.lm")) {
                                        code
                                    } else {
                                        return None;
                                    },
                                )?)?,
                            ));
                            Some(Type::Text(format!("Logged in project: {path}")))
                        } else {
                            None
                        }
                    })),
                ),
                (
                    "logout".to_string(),
                    Type::Function(Function::BuiltIn(|_, engine| {
                        engine.project = None;
                        Some(Type::Text("Logged out current project".to_string()))
                    })),
                ),
                (
                    "run".to_string(),
                    Type::Function(Function::BuiltIn(|_, engine| {
                        if let Some((project_path, _)) = engine.project.clone() {
                            if let Ok(dir) = read_dir(Path::new(&format!("{project_path}/lib"))) {
                                for entry in dir {
                                    if let Ok(entry) = entry {
                                        let lib_file = entry.file_name();
                                        if let Ok(code) = read_to_string(Path::new(&format!(
                                            "{project_path}/lib/{}",
                                            lib_file.to_str()?
                                        ))) {
                                            engine.eval(Engine::parse(code)?);
                                        } else {
                                            return None;
                                        }
                                    } else {
                                        return None;
                                    }
                                }
                            } else {
                                return None;
                            }
                            if let Ok(code) =
                                read_to_string(Path::new(&format!("{project_path}/src/main.lm")))
                            {
                                engine.eval(Engine::parse(code)?)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })),
                ),
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
            result = code.eval(self)?
        }
        Some(result)
    }
}

#[derive(Debug, Clone)]
enum Statement {
    Print(Vec<Expr>),
    Let(String, Option<Signature>, Expr),
    If(Expr, Expr, Option<Expr>),
    Match(Expr, Vec<(Vec<Expr>, Expr)>),
    For(String, Expr, Expr),
    While(Expr, Expr),
    Fault,
    Return(Expr),
}

impl Statement {
    fn eval(&self, engine: &mut Engine) -> Option<Type> {
        Some(match self {
            Statement::Print(expr) => {
                for i in expr {
                    print!(
                        "{}",
                        match i.eval(engine)? {
                            Type::Text(text) => text,
                            other => other.get_symbol(),
                        }
                    );
                }
                println!();
                io::stdout().flush().unwrap();
                Type::Null
            }
            Statement::Let(name, sig, expr) => {
                let val = expr.eval(engine)?;
                if let Some(sig) = sig {
                    if val.get_type().format() != sig.format() {
                        return None;
                    }
                }
                if name != "_" {
                    engine.env.insert(name.to_owned(), val.clone());
                }
                val
            }
            Statement::If(expr, then, r#else) => {
                if let Some(it) = expr.eval(engine) {
                    engine.env.insert("it".to_string(), it);
                    then.eval(engine)?
                } else {
                    if let Some(r#else) = r#else {
                        r#else.eval(engine)?
                    } else {
                        Type::Null
                    }
                }
            }
            Statement::Match(expr, conds) => {
                let expr = expr.eval(engine)?;
                for (conds, value) in conds {
                    for cond in conds {
                        let cond = cond.eval(engine)?;
                        if expr.is_match(&cond) {
                            return value.eval(engine);
                        }
                    }
                }
                return None;
            }
            Statement::For(counter, expr, code) => {
                let mut result = Type::Null;
                for i in expr.eval(engine)?.get_list() {
                    if counter != "_" {
                        engine.env.insert(counter.clone(), i);
                    }
                    result = code.eval(engine)?;
                }
                result
            }
            Statement::While(expr, code) => {
                let mut result = Type::Null;
                while let Some(it) = expr.eval(engine) {
                    engine.env.insert("it".to_string(), it);
                    result = code.eval(engine)?;
                }
                result
            }
            Statement::Fault => return None,
            Statement::Return(expr) => expr.eval(engine)?,
        })
    }

    fn parse(code: String) -> Option<Statement> {
        let code = code.trim().to_string();
        if code.starts_with("print") {
            let mut exprs = vec![];
            for i in tokenize(code["print".len()..].to_string(), vec![','])? {
                exprs.push(Expr::parse(i)?)
            }
            Some(Statement::Print(exprs))
        } else if code.starts_with("let") {
            let code = code["let".len()..].to_string();
            let (name, code) = code.split_once("=")?;
            if let Some((name, sig)) = name.split_once(":") {
                Some(Statement::Let(
                    name.trim().to_string(),
                    Some(Signature::parse(sig.trim().to_string())?),
                    Expr::parse(code.to_string())?,
                ))
            } else {
                Some(Statement::Let(
                    name.trim().to_string(),
                    None,
                    Expr::parse(code.to_string())?,
                ))
            }
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
            let mut body = vec![];
            for i in tokens {
                let tokens = tokenize(i, SPACE.to_vec())?;
                let pos = tokens.iter().position(|i| i == "=>")?;
                let mut cond = vec![];
                for i in tokenize(tokens.get(..pos)?.join(&SPACE[0].to_string()), vec!['|'])? {
                    cond.push(Expr::parse(i.to_string())?)
                }
                body.push((
                    cond,
                    Expr::parse(tokens.get(pos + 1..)?.join(&SPACE[0].to_string()))?,
                ))
            }
            Some(Statement::Match(expr, body))
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
        } else if code == "fault" {
            Some(Statement::Fault)
        } else if code.starts_with("return") {
            let code = code["return".len()..].to_string();
            Some(Statement::Return(Expr::parse(code.to_string())?))
        } else {
            Some(Statement::Return(Expr::parse(code.to_string())?))
        }
    }

    fn format(&self) -> String {
        match self {
            Statement::Print(exprs) => format!(
                "print {}",
                exprs
                    .iter()
                    .map(|i| i.format())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Statement::Let(name, Some(sig), val) => {
                format!("let {name}: {} = {}", sig.format(), val.format())
            }
            Statement::Let(name, None, val) => {
                format!("let {name} = {}", val.format())
            }
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
            Statement::For(counter, iterator, code) => {
                format!("for {counter} in {} {}", iterator.format(), code.format())
            }
            Statement::While(cond, code) => {
                format!("while {} {}", cond.format(), code.format())
            }
            Statement::Fault => "fault".to_string(),
            Statement::Return(expr) => format!("{}", expr.format()),
        }
    }
}

#[derive(Debug, Clone)]
enum Expr {
    Derefer(Box<Expr>),
    Infix(Box<Operator>),
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
                    result.insert(k.eval(engine)?.get_text()?, x.eval(engine)?);
                }
                Type::Struct(result)
            }
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
            Expr::Block(Engine::parse(token.clone())?)
        } else if token.starts_with("@{") && token.ends_with('}') {
            let token = token.get(2..token.len() - 1)?.to_string();
            let mut result = Vec::new();
            for i in tokenize(token.clone(), vec![','])? {
                let splited = tokenize(i, vec![':'])?;
                result.push((
                    Expr::parse(splited.get(0)?.to_string())?,
                    Expr::parse(splited.get(1)?.to_string())?,
                ));
            }
            Expr::Struct(result)
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
        } else if token.starts_with("fn(") && token.contains("->") && token.ends_with(")") {
            let token = token.replacen("fn(", "", 1);
            let token = token.get(..token.len() - 1)?.to_string();
            let (args, body) = token.split_once("->")?;
            let mut args: Vec<&str> = args.split(",").collect();
            args.reverse();
            let mut func = Expr::Value(Type::Function(Function::UserDefined(
                args.first()?.trim().to_string(),
                Box::new(Expr::parse(body.to_string())?),
            )));
            for arg in args.get(1..)? {
                func = Expr::Value(Type::Function(Function::UserDefined(
                    arg.trim().to_string(),
                    Box::new(func),
                )));
            }
            func
        } else if token.contains('(') && token.ends_with(')') {
            let token = token.get(..token.len() - 1)?.to_string();
            let (name, args) = token.split_once("(")?;
            let args = tokenize(args.to_string(), vec![','])?;
            let mut call = Expr::Infix(Box::new(Operator::Apply(
                Expr::parse(name.to_string())?,
                Expr::parse(args.first()?.to_string())?,
            )));
            for arg in args.get(1..)? {
                call = Expr::Infix(Box::new(Operator::Apply(
                    call,
                    Expr::parse(arg.to_string())?,
                )));
            }
            call
        } else if token.starts_with("&") {
            let token = token.replacen("&", "", 1);
            Expr::Value(Type::Refer(token))
        } else if token.starts_with("*") {
            let token = token.replacen("*", "", 1);
            Expr::Derefer(Box::new(Expr::parse(token)?))
        } else if token == "null" {
            Expr::Value(Type::Null)
        } else {
            Expr::Value(Type::Symbol(token))
        };

        if let Some(operator) = token_list
            .len()
            .checked_sub(2)
            .and_then(|idx| token_list.get(idx))
        {
            let has_lhs = |token_list: Vec<String>| {
                Some(Expr::parse(
                    token_list
                        .get(..token_list.len() - 2)?
                        .join(&SPACE[0].to_string()),
                )?)
            };
            Some(Expr::Infix(Box::new(match operator.as_str() {
                "+" => Operator::Add(has_lhs(token_list)?, token),
                "-" => Operator::Sub(has_lhs(token_list)?, token),
                "*" => Operator::Mul(has_lhs(token_list)?, token),
                "/" => Operator::Div(has_lhs(token_list)?, token),
                "%" => Operator::Mod(has_lhs(token_list)?, token),
                "^" => Operator::Pow(has_lhs(token_list)?, token),
                "==" => Operator::Equal(has_lhs(token_list)?, token),
                "!=" => Operator::NotEq(has_lhs(token_list)?, token),
                "<" => Operator::LessThan(has_lhs(token_list)?, token),
                "<=" => Operator::LessThanEq(has_lhs(token_list)?, token),
                ">" => Operator::GreaterThan(has_lhs(token_list)?, token),
                ">=" => Operator::GreaterThanEq(has_lhs(token_list)?, token),
                "&" => Operator::And(has_lhs(token_list)?, token),
                "|" => Operator::Or(has_lhs(token_list)?, token),
                "!" => Operator::Not(token),
                "::" => Operator::Access(has_lhs(token_list)?, token),
                "as" => Operator::As(has_lhs(token_list)?, token),
                ":=" => Operator::Assign(has_lhs(token_list)?, token),
                "|>" => Operator::PipeLine(has_lhs(token_list)?, token),
                operator => {
                    if operator.starts_with("`") && operator.ends_with("`") {
                        let operator = operator[1..operator.len() - 1].to_string();
                        Operator::Apply(
                            Expr::Infix(Box::new(Operator::Apply(
                                Expr::parse(operator)?,
                                has_lhs(token_list)?,
                            ))),
                            token,
                        )
                    } else {
                        Operator::Apply(
                            Expr::parse(
                                token_list
                                    .get(..token_list.len() - 1)?
                                    .join(&SPACE[0].to_string()),
                            )?,
                            token,
                        )
                    }
                }
            })))
        } else {
            Some(token)
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
                "@{{ {} }}",
                st.iter()
                    .map(|(k, x)| format!("{}: {}", k.format(), x.format()))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Expr::Derefer(to) => format!("*{}", to.format()),
        }
    }

    /// Replacing constant arguments when apply function
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
            Expr::Infix(infix) => Expr::Infix(Box::new(match *infix.clone() {
                Operator::Add(lhs, rhs) => {
                    Operator::Add(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::Sub(lhs, rhs) => {
                    Operator::Sub(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::Mul(lhs, rhs) => {
                    Operator::Mul(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::Div(lhs, rhs) => {
                    Operator::Div(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::Mod(lhs, rhs) => {
                    Operator::Mod(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::Pow(lhs, rhs) => {
                    Operator::Pow(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::Equal(lhs, rhs) => {
                    Operator::Equal(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::NotEq(lhs, rhs) => {
                    Operator::NotEq(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::LessThan(lhs, rhs) => {
                    Operator::LessThan(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::LessThanEq(lhs, rhs) => {
                    Operator::LessThanEq(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::GreaterThan(lhs, rhs) => {
                    Operator::GreaterThan(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::GreaterThanEq(lhs, rhs) => {
                    Operator::GreaterThanEq(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::And(lhs, rhs) => {
                    Operator::And(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::Or(lhs, rhs) => {
                    Operator::Or(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::Not(val) => Operator::Not(val.replace(from, to)),
                Operator::Access(lhs, rhs) => {
                    Operator::Access(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::As(lhs, rhs) => {
                    Operator::As(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::Apply(lhs, rhs) => {
                    Operator::Apply(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::Assign(lhs, rhs) => {
                    Operator::Assign(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::PipeLine(lhs, rhs) => {
                    Operator::PipeLine(lhs.replace(from, to), rhs.replace(from, to))
                }
            })),
            Expr::Block(block) => Expr::Block(
                block
                    .iter()
                    .map(|i| match i {
                        Statement::Print(vals) => {
                            Statement::Print(vals.iter().map(|j| j.replace(from, to)).collect())
                        }
                        Statement::Let(name, sig, val) => {
                            Statement::Let(name.clone(), sig.clone(), val.replace(from, to))
                        }
                        Statement::If(cond, then, r#else) => Statement::If(
                            cond.replace(from, to),
                            then.replace(from, to),
                            r#else.clone().and_then(|j| Some(j.replace(from, to))),
                        ),
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
                        Statement::For(counter, iter, code) => Statement::For(
                            counter.clone(),
                            iter.replace(from, to),
                            code.replace(from, to),
                        ),
                        Statement::While(cond, code) => {
                            Statement::While(cond.replace(from, to), code.replace(from, to))
                        }
                        Statement::Fault => Statement::Fault,
                        Statement::Return(val) => Statement::Return(val.replace(from, to)),
                    })
                    .collect(),
            ),
            Expr::Value(Type::Function(Function::UserDefined(arg, func))) => {
                Expr::Value(Type::Function(Function::UserDefined(
                    arg.to_string(),
                    // Protect from duplicate replacing
                    if from.format() != *arg {
                        Box::new(func.replace(from, to))
                    } else {
                        func.clone()
                    },
                )))
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
enum Operator {
    Add(Expr, Expr),
    Sub(Expr, Expr),
    Mul(Expr, Expr),
    Div(Expr, Expr),
    Mod(Expr, Expr),
    Pow(Expr, Expr),
    Equal(Expr, Expr),
    NotEq(Expr, Expr),
    LessThan(Expr, Expr),
    LessThanEq(Expr, Expr),
    GreaterThan(Expr, Expr),
    GreaterThanEq(Expr, Expr),
    And(Expr, Expr),
    Or(Expr, Expr),
    Not(Expr),
    Access(Expr, Expr),
    As(Expr, Expr),
    Apply(Expr, Expr),
    Assign(Expr, Expr),
    PipeLine(Expr, Expr),
}

impl Operator {
    fn eval(&self, engine: &mut Engine) -> Option<Type> {
        Some(match self {
            Operator::Add(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                if let (Type::Number(lhs), Type::Number(rhs)) = (&lhs, &rhs) {
                    Type::Number(lhs + rhs)
                } else if let (Type::Text(lhs), Type::Text(rhs)) = (&lhs, &rhs) {
                    Type::Text(lhs.clone() + &rhs)
                } else if let (Type::List(lhs), Type::List(rhs)) = (&lhs, &rhs) {
                    Type::List([lhs.clone(), rhs.clone()].concat())
                } else if let (Type::Struct(mut lhs), Type::Struct(rhs)) = (lhs.clone(), &rhs) {
                    lhs.extend(rhs.clone());
                    Type::Struct(lhs)
                } else {
                    return None;
                }
            }
            Operator::Sub(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                if let (Type::Number(lhs), Type::Number(rhs)) = (&lhs, &rhs) {
                    Type::Number(lhs - rhs)
                } else if let (Type::Text(lhs), Type::Text(rhs)) = (&lhs, &rhs) {
                    Type::Text(lhs.replacen(rhs, "", 1))
                } else if let (Type::List(mut lhs), Type::List(rhs)) = (lhs.clone(), &rhs) {
                    let first_index = lhs.windows(rhs.len()).position(|i| {
                        i.iter().map(|j| j.get_symbol()).collect::<Vec<_>>()
                            == rhs.iter().map(|j| j.get_symbol()).collect::<Vec<_>>()
                    })?;
                    for _ in 0..rhs.len() {
                        lhs.remove(first_index);
                    }
                    Type::List(lhs)
                } else if let (Type::List(mut lhs), Type::Number(rhs)) = (lhs.clone(), &rhs) {
                    lhs.remove(rhs.clone() as usize);
                    Type::List(lhs)
                } else if let (Type::Text(mut lhs), Type::Number(rhs)) = (lhs, rhs) {
                    lhs.remove(rhs as usize);
                    Type::Text(lhs)
                } else {
                    return None;
                }
            }
            Operator::Mul(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                if let (Type::Number(lhs), Type::Number(rhs)) = (&lhs, &rhs) {
                    Type::Number(lhs * rhs)
                } else if let (Type::Text(lhs), Type::Number(rhs)) = (&lhs, &rhs) {
                    Type::Text(lhs.repeat(*rhs as usize))
                } else if let (Type::List(lhs), Type::Number(rhs)) = (lhs, rhs) {
                    Type::List((0..rhs as usize).flat_map(|_| lhs.clone()).collect())
                } else {
                    return None;
                }
            }
            Operator::Div(lhs, rhs) => {
                Type::Number(lhs.eval(engine)?.get_number()? / rhs.eval(engine)?.get_number()?)
            }
            Operator::Mod(lhs, rhs) => {
                Type::Number(lhs.eval(engine)?.get_number()? % rhs.eval(engine)?.get_number()?)
            }
            Operator::Pow(lhs, rhs) => Type::Number(
                lhs.eval(engine)?
                    .get_number()?
                    .powf(rhs.eval(engine)?.get_number()?),
            ),
            Operator::Equal(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_symbol() == rhs.get_symbol() {
                    rhs
                } else {
                    return None;
                }
            }
            Operator::NotEq(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_symbol() != rhs.get_symbol() {
                    rhs
                } else {
                    return None;
                }
            }
            Operator::LessThan(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number() < rhs.get_number() {
                    rhs
                } else {
                    return None;
                }
            }
            Operator::LessThanEq(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number() <= rhs.get_number() {
                    rhs
                } else {
                    return None;
                }
            }
            Operator::GreaterThan(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number() > rhs.get_number() {
                    rhs
                } else {
                    return None;
                }
            }
            Operator::GreaterThanEq(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number() >= rhs.get_number() {
                    rhs
                } else {
                    return None;
                }
            }
            Operator::And(lhs, rhs) => {
                let rhs = rhs.eval(engine);
                if lhs.eval(engine).is_some() && rhs.is_some() {
                    rhs?
                } else {
                    return None;
                }
            }
            Operator::Or(lhs, rhs) => {
                let lhs = lhs.eval(engine);
                let rhs = rhs.eval(engine);
                if lhs.is_some() || rhs.is_some() {
                    rhs.unwrap_or(lhs?)
                } else {
                    return None;
                }
            }
            Operator::Not(val) => {
                let val = val.eval(engine);
                if val.is_some() {
                    return None;
                } else {
                    Type::Null
                }
            }
            Operator::Access(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                if let (Type::List(list), Type::Number(index)) = (lhs.clone(), rhs.clone()) {
                    list.get(index as usize)?.clone()
                } else if let (Type::Text(text), Type::Number(index)) = (lhs.clone(), rhs.clone()) {
                    Type::Text(
                        text.chars()
                            .collect::<Vec<char>>()
                            .get(index as usize)?
                            .to_string(),
                    )
                } else if let (Type::Struct(st), Type::Text(index)) = (lhs.clone(), rhs.clone()) {
                    st.get(&index)?.clone()
                } else {
                    return None;
                }
            }
            Operator::As(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                match rhs.get_signature()? {
                    Signature::Number => Type::Number(lhs.get_number()?),
                    Signature::Symbol => Type::Symbol(lhs.get_symbol()),
                    Signature::Text => Type::Text(lhs.get_text()?),
                    Signature::List => Type::List(lhs.get_list()),
                    Signature::Function => Type::Function(lhs.get_function()?),
                    Signature::Refer => Type::Refer(lhs.get_symbol()),
                    Signature::Struct => Type::Struct(lhs.get_struct()?),
                    Signature::Signature => Type::Signature(Signature::parse(lhs.get_symbol())?),
                }
            }
            Operator::Apply(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                match lhs.get_function()? {
                    Function::BuiltIn(func) => func(rhs, engine)?,
                    Function::UserDefined(parameter, code) => {
                        let code =
                            code.replace(&Expr::Value(Type::Symbol(parameter)), &Expr::Value(rhs));
                        code.eval(&mut engine.clone())?
                    }
                }
            }
            Operator::Assign(lhs, rhs) => {
                let name = lhs.eval(engine)?.get_text()?;
                Statement::Let(name, None, rhs.to_owned()).eval(engine)?
            }
            Operator::PipeLine(lhs, rhs) => {
                Operator::Apply(rhs.to_owned(), lhs.to_owned()).eval(engine)?
            }
        })
    }

    fn format(&self) -> String {
        match self {
            Operator::Add(lhs, rhs) => format!("{} + {}", lhs.format(), rhs.format()),
            Operator::Sub(lhs, rhs) => format!("{} - {}", lhs.format(), rhs.format()),
            Operator::Mul(lhs, rhs) => format!("{} * {}", lhs.format(), rhs.format()),
            Operator::Div(lhs, rhs) => format!("{} / {}", lhs.format(), rhs.format()),
            Operator::Mod(lhs, rhs) => format!("{} % {}", lhs.format(), rhs.format()),
            Operator::Pow(lhs, rhs) => format!("{} ^ {}", lhs.format(), rhs.format()),
            Operator::Equal(lhs, rhs) => format!("{} == {}", lhs.format(), rhs.format()),
            Operator::NotEq(lhs, rhs) => format!("{} != {}", lhs.format(), rhs.format()),
            Operator::LessThan(lhs, rhs) => format!("{} < {}", lhs.format(), rhs.format()),
            Operator::LessThanEq(lhs, rhs) => format!("{} <= {}", lhs.format(), rhs.format()),
            Operator::GreaterThan(lhs, rhs) => format!("{} > {}", lhs.format(), rhs.format()),
            Operator::GreaterThanEq(lhs, rhs) => {
                format!("{} >= {}", lhs.format(), rhs.format())
            }
            Operator::And(lhs, rhs) => format!("{} & {}", lhs.format(), rhs.format()),
            Operator::Or(lhs, rhs) => format!("{} | {}", lhs.format(), rhs.format()),
            Operator::Not(val) => format!("! {}", val.format()),
            Operator::Access(lhs, rhs) => format!("{} :: {}", lhs.format(), rhs.format()),
            Operator::As(lhs, rhs) => format!("{} as {}", lhs.format(), rhs.format()),
            Operator::Assign(lhs, rhs) => format!("{} := {}", lhs.format(), rhs.format()),
            Operator::PipeLine(lhs, rhs) => format!("{} |> {}", lhs.format(), rhs.format()),
            Operator::Apply(lhs, rhs) => format!("{} {}", lhs.format(), rhs.format()),
        }
        .to_string()
    }
}

#[derive(Debug, Clone)]
enum Type {
    Number(f64),
    Symbol(String),
    Refer(String),
    Text(String),
    List(Vec<Type>),
    Function(Function),
    Signature(Signature),
    Struct(HashMap<String, Type>),
    Null,
}

impl Type {
    fn get_symbol(&self) -> String {
        match self {
            Type::Symbol(s) => s.to_string(),
            Type::Text(s) => format!("\"{s}\""),
            Type::Number(n) => n.to_string(),
            Type::Null => "null".to_string(),
            Type::Function(Function::BuiltIn(obj)) => format!("λx.{obj:?}"),
            Type::Function(Function::UserDefined(arg, code)) => {
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
            Type::Refer(to) => format!("&{to}"),
            Type::Struct(val) => format!(
                "@{{ {} }}",
                val.iter()
                    .map(|(k, v)| format!("\"{k}\": {}", v.get_symbol()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

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

    fn get_text(&self) -> Option<String> {
        match self {
            Type::Symbol(s) | Type::Text(s) => Some(s.to_string()),
            Type::Number(n) => Some(n.to_string()),
            Type::Signature(s) => Some(s.format()),
            _ => None,
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

    fn get_type(&self) -> Signature {
        match self {
            Type::Number(_) => Signature::Number,
            Type::Text(_) => Signature::Text,
            Type::Refer(_) => Signature::Refer,
            Type::Symbol(_) => Signature::Symbol,
            Type::List(_) => Signature::List,
            Type::Signature(_) => Signature::Signature,
            Type::Function(_) => Signature::Function,
            Type::Struct(_) => Signature::Struct,
            Type::Null => Signature::Symbol,
        }
    }

    fn get_signature(&self) -> Option<Signature> {
        if let Type::Signature(sig) = self {
            Some(sig.clone())
        } else {
            None
        }
    }

    fn get_struct(&self) -> Option<HashMap<String, Type>> {
        if let Type::Struct(val) = self {
            Some(val.clone())
        } else {
            None
        }
    }

    fn is_match(&self, pattern: &Type) -> bool {
        if let (Type::List(list), Type::List(pats)) = (self, pattern) {
            if list.len() != pats.len() {
                return false;
            }
            for (elm, pat) in list.iter().zip(pats) {
                if !elm.is_match(pat) {
                    return false;
                }
            }
            true
        } else if let (Type::Struct(strct), Type::Struct(pats)) = (self, pattern) {
            if strct.len() != pats.len() {
                return false;
            }
            for (elm, pat) in strct.iter().zip(pats) {
                if elm.0 != pat.0 || !elm.1.is_match(pat.1) {
                    return false;
                }
            }
            true
        } else {
            if pattern.get_symbol() == "_" {
                true
            } else {
                self.get_symbol() == pattern.get_symbol()
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Signature {
    Number,
    Symbol,
    Refer,
    Text,
    List,
    Function,
    Signature,
    Struct,
}

impl Signature {
    fn parse(token: String) -> Option<Signature> {
        Some(if token == "number" {
            Signature::Number
        } else if token == "symbol" {
            Signature::Symbol
        } else if token == "text" {
            Signature::Text
        } else if token == "refer" {
            Signature::Refer
        } else if token == "list" {
            Signature::List
        } else if token == "function" {
            Signature::Function
        } else if token == "signature" {
            Signature::Signature
        } else if token == "struct" {
            Signature::Struct
        } else {
            return None;
        })
    }

    fn format(&self) -> String {
        format!("{self:?}").to_lowercase()
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
