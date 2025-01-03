use clap::Parser;
use colored::*;
use reqwest::blocking;
use rustyline::{error::ReadlineError, DefaultEditor};
use std::{
    collections::HashMap,
    f64::consts::PI,
    fs::{read_to_string, File},
    io::{self, Write},
    process::exit,
};
use thiserror::Error;

const VERSION: &str = "0.3.2";
const SPACE: [char; 5] = [' ', '　', '\n', '\t', '\r'];

macro_rules! ok {
    ($option_value: expr, $fault_kind: expr) => {
        if let Some(ok) = $option_value {
            Ok(ok)
        } else {
            Err($fault_kind)
        }
    };
    ($option_value: expr) => {
        if let Some(ok) = $option_value {
            Ok(ok)
        } else {
            Err(Fault::Syntax)
        }
    };
}

#[derive(Parser)]
#[command(
    name = "Lamuta",version = VERSION,
    about = "A functional programming language that can write lambda calculus formula as they are"
)]
struct Cli {
    /// Source file to evaluate
    #[arg(index = 1)]
    file: Option<String>,

    /// Command-line arguments to pass the script
    #[arg(index = 2, value_name="ARGS", num_args = 0..)]
    args_position: Option<Vec<String>>,

    /// Optional command-line arguments
    #[arg(short='a', long="args", value_name="ARGS", num_args = 0..)]
    args_option: Option<Vec<String>>,
}

fn main() {
    let cli = Cli::parse();
    let mut engine = Engine::new();

    if let (Some(args), _) | (_, Some(args)) = (cli.args_position, cli.args_option) {
        engine.env.insert(
            "cmdLineArgs".to_string(),
            Type::List(args.iter().map(|i| Type::Text(i.to_owned())).collect()),
        );
    }

    if let Some(file) = cli.file {
        if let Err(e) = Operator::Apply(
            Expr::Value(Type::Symbol("load".to_string())),
            Expr::Value(Type::Text(file)),
        )
        .eval(&mut engine)
        {
            eprintln!("Fault {e}")
        }
    } else {
        println!("{title} {VERSION}", title = "Lamuta".blue().bold());
        let mut rl = DefaultEditor::new().unwrap();
        let mut session = 1;

        loop {
            match rl.readline(&format!("[{session:0>3}]> ")) {
                Ok(code) => {
                    match Engine::parse(code) {
                        Ok(ast) => match engine.eval(ast) {
                            Ok(result) => {
                                println!("{navi} {}", result.get_symbol(), navi = "=>".green())
                            }
                            Err(e) => println!("{navi} Fault: {e}", navi = "=>".red()),
                        },
                        Err(e) => println!("{navi} Syntax Error: {e}", navi = "=>".red()),
                    }
                    session += 1;
                }
                Err(ReadlineError::Interrupted) => println!("^C"),
                Err(ReadlineError::Eof) => println!("^D"),
                _ => {}
            }
        }
    }
}

type Scope = HashMap<String, Type>;
type Program = Vec<Statement>;
#[derive(Debug, Clone)]
struct Engine {
    env: Scope,
    protect: Vec<String>,
}

impl Engine {
    fn new() -> Engine {
        Engine {
            protect: vec![
                "type".to_string(),
                "env".to_string(),
                "free".to_string(),
                "eval".to_string(),
                "alphaConvert".to_string(),
                "input".to_string(),
                "range".to_string(),
                "exit".to_string(),
                "load".to_string(),
                "save".to_string(),
                "pi".to_string(),
                "cmdLineArgs".to_string(),
            ],
            env: HashMap::from([
                (
                    "type".to_string(),
                    Type::Function(Function::BuiltIn(|expr, _| {
                        Ok(Type::Signature(expr.get_type()))
                    })),
                ),
                (
                    "env".to_string(),
                    Type::Function(Function::BuiltIn(|_, engine| {
                        Ok(Type::Struct(engine.env.clone()))
                    })),
                ),
                (
                    "free".to_string(),
                    Type::Function(Function::BuiltIn(|name, engine| {
                        let name = &name.get_refer()?;
                        if engine.protect.contains(name) {
                            return Err(Fault::AccessDenied);
                        }
                        engine.env.remove(name);
                        Ok(Type::Null)
                    })),
                ),
                (
                    "eval".to_string(),
                    Type::Function(Function::BuiltIn(|args, engine| {
                        let args = args.get_list();
                        let code = ok!(args.get(0))?.get_text()?;
                        if let Some(env) = args.get(1) {
                            let mut engine = engine.clone();
                            engine.env = env.get_struct()?;
                            engine.eval(Engine::parse(code)?)
                        } else {
                            engine.eval(Engine::parse(code)?)
                        }
                    })),
                ),
                (
                    "alphaConvert".to_string(),
                    Type::Function(Function::BuiltIn(|args, _| {
                        let args = args.get_list();
                        let func = ok!(args.get(0), Fault::MissMatchArgLen)?;
                        let new_name = ok!(args.get(1))?.get_text()?;
                        let Type::Function(Function::UserDefined(arg, body)) = func else {
                            return Err(Fault::Type {
                                value: func.to_owned(),
                                annotate: Signature::Function,
                            });
                        };
                        Ok(Type::Function(Function::UserDefined(
                            new_name.clone(),
                            Box::new(body.replace(
                                &Expr::Value(Type::Symbol(arg.to_owned())),
                                &Expr::Value(Type::Symbol(new_name)),
                            )),
                        )))
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
                            Ok(Type::Text(buffer.trim().to_string()))
                        } else {
                            Err(Fault::IO)
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
                            Ok(Type::List(range))
                        } else if params.len() == 2 {
                            let mut range: Vec<Type> = vec![];
                            let mut current: f64 = params[0].get_number()?;
                            while current < params[1].get_number()? {
                                range.push(Type::Number(current));
                                current += 1.0;
                            }
                            Ok(Type::List(range))
                        } else if params.len() == 3 {
                            let mut range: Vec<Type> = vec![];
                            let mut current: f64 = params[0].get_number()?;
                            while current < params[1].get_number()? {
                                range.push(Type::Number(current));
                                current += params[2].get_number()?;
                            }
                            Ok(Type::List(range))
                        } else {
                            Err(Fault::MissMatchArgLen)
                        }
                    })),
                ),
                (
                    "exit".to_string(),
                    Type::Function(Function::BuiltIn(|arg, _| exit(arg.get_number()? as i32))),
                ),
                (
                    "load".to_string(),
                    Type::Function(Function::BuiltIn(|expr, engine| {
                        let name = expr.get_text()?;
                        if let Ok(module) = read_to_string(&name) {
                            engine.eval(Engine::parse(module)?)
                        } else if let Ok(module) = blocking::get(name) {
                            if let Ok(code) = module.text() {
                                engine.eval(Engine::parse(code)?)
                            } else {
                                Err(Fault::IO)
                            }
                        } else {
                            Err(Fault::IO)
                        }
                    })),
                ),
                (
                    "save".to_string(),
                    Type::Function(Function::BuiltIn(|arg, engine| {
                        let mut render = String::new();
                        for (k, v) in &engine.env {
                            if !engine.protect.contains(k) {
                                render += &format!("let {k} = {};\n", v.get_symbol());
                            }
                        }
                        if let Ok(mut file) = File::create(arg.get_text()?) {
                            if file.write_all(render.as_bytes()).is_ok() {
                                Ok(Type::Text("Saved environment".to_string()))
                            } else {
                                Err(Fault::IO)
                            }
                        } else {
                            Err(Fault::IO)
                        }
                    })),
                ),
                ("pi".to_string(), Type::Number(PI)),
            ]),
        }
    }

    fn parse(source: String) -> Result<Program, Fault> {
        let mut program: Program = Vec::new();
        for line in tokenize(source, vec![';'])? {
            let line = line.trim().to_string();
            // Ignore empty line and comment
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            program.push(Statement::parse(line)?);
        }
        Ok(program)
    }

    fn eval(&mut self, program: Program) -> Result<Type, Fault> {
        let mut result = Type::Null;
        for code in program {
            result = code.eval(self)?
        }
        Ok(result)
    }
}

#[derive(Debug, Clone)]
enum Statement {
    Print(Vec<Expr>),
    Let(String, bool, Option<Signature>, Expr),
    If(Expr, Expr, Option<Expr>),
    Match(Expr, Vec<(Vec<Expr>, Expr)>),
    For(String, Expr, Expr),
    While(Expr, Expr),
    Fault,
    Return(Expr),
}

impl Statement {
    fn eval(&self, engine: &mut Engine) -> Result<Type, Fault> {
        Ok(match self {
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
                io::stdout().flush().unwrap();
                Type::Null
            }
            Statement::Let(name, protect, sig, expr) => {
                let val = expr.eval(engine)?;
                if engine.protect.contains(name) {
                    return Err(Fault::AccessDenied);
                }
                if let Some(sig) = sig {
                    if val.get_type().format() != sig.format() {
                        return Err(Fault::Type {
                            value: val,
                            annotate: sig.to_owned(),
                        });
                    }
                }
                if name != "_" {
                    engine.env.insert(name.to_owned(), val.clone());
                    if *protect {
                        engine.protect.push(name.to_owned());
                    }
                }
                val
            }
            Statement::If(expr, then, r#else) => {
                if let Ok(it) = expr.eval(engine) {
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
                return Err(Fault::Syntax);
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
                while let Ok(it) = expr.eval(engine) {
                    engine.env.insert("it".to_string(), it);
                    result = code.eval(engine)?;
                }
                result
            }
            Statement::Fault => return Err(Fault::Syntax),
            Statement::Return(expr) => expr.eval(engine)?,
        })
    }

    fn parse(code: String) -> Result<Statement, Fault> {
        let code = code.trim().to_string();
        if code.starts_with("print") {
            let mut exprs = vec![];
            for i in tokenize(code["print".len()..].to_string(), vec![','])? {
                exprs.push(Expr::parse(i)?)
            }
            Ok(Statement::Print(exprs))
        } else if code.starts_with("let") {
            let code = code["let".len()..].to_string();
            let (name, code) = ok!(code.split_once("="))?;
            if let Some((name, sig)) = name.split_once(":") {
                Ok(Statement::Let(
                    name.trim().to_string(),
                    false,
                    Some(ok!(Signature::parse(sig.trim().to_string()))?),
                    Expr::parse(code.to_string())?,
                ))
            } else {
                Ok(Statement::Let(
                    name.trim().to_string(),
                    false,
                    None,
                    Expr::parse(code.to_string())?,
                ))
            }
        } else if code.starts_with("const") {
            let code = code["const".len()..].to_string();
            let (name, code) = ok!(code.split_once("="))?;
            let (name, sig) = ok!(name.split_once(":"))?;
            Ok(Statement::Let(
                name.trim().to_string(),
                true,
                Some(ok!(Signature::parse(sig.trim().to_string()))?),
                Expr::parse(code.to_string())?,
            ))
        } else if code.starts_with("if") {
            let code = code["if".len()..].to_string();
            let code = tokenize(code, SPACE.to_vec())?;
            if let Some(pos) = code.iter().position(|i| i == "else") {
                Ok(Statement::If(
                    Expr::parse(ok!(code.get(0))?.to_string())?,
                    Expr::parse(ok!(code.get(1..pos))?.join(&SPACE[0].to_string()))?,
                    Some(Expr::parse(
                        ok!(code.get(pos + 1..))?.join(&SPACE[0].to_string()),
                    )?),
                ))
            } else {
                Ok(Statement::If(
                    Expr::parse(ok!(code.get(0))?.to_string())?,
                    Expr::parse(ok!(code.get(1..))?.join(&SPACE[0].to_string()))?,
                    None,
                ))
            }
        } else if code.starts_with("match") {
            let code = code["match".len()..].to_string();
            let tokens = tokenize(code, SPACE.to_vec())?;
            let expr = Expr::parse(ok!(tokens.get(0))?.to_string())?;
            let tokens = tokenize(
                ok!(tokens.get(1))?[1..ok!(tokens.get(1))?.len() - 1].to_string(),
                vec![','],
            )?;
            let mut body = vec![];
            for i in tokens {
                let tokens = tokenize(i, SPACE.to_vec())?;
                let pos = ok!(tokens.iter().position(|i| i == "=>"))?;
                let mut cond = vec![];
                for i in tokenize(
                    ok!(tokens.get(..pos))?.join(&SPACE[0].to_string()),
                    vec!['|'],
                )? {
                    cond.push(Expr::parse(i.to_string())?)
                }
                body.push((
                    cond,
                    Expr::parse(ok!(tokens.get(pos + 1..))?.join(&SPACE[0].to_string()))?,
                ))
            }
            Ok(Statement::Match(expr, body))
        } else if code.starts_with("for") {
            let code = code["for".len()..].to_string();
            let code = tokenize(code, SPACE.to_vec())?;
            if code.get(1).and_then(|x| Some(x == "in")).unwrap_or(false) {
                Ok(Statement::For(
                    ok!(code.get(0))?.to_string(),
                    Expr::parse(ok!(code.get(2))?.to_string())?,
                    Expr::parse(ok!(code.get(3))?.to_string())?,
                ))
            } else {
                Err(Fault::Syntax)
            }
        } else if code.starts_with("while") {
            let code = code["while".len()..].to_string();
            let code = tokenize(code, SPACE.to_vec())?;
            Ok(Statement::While(
                Expr::parse(ok!(code.get(0))?.to_string())?,
                Expr::parse(ok!(code.get(1))?.to_string())?,
            ))
        } else if code == "fault" {
            Ok(Statement::Fault)
        } else {
            Ok(Statement::Return(Expr::parse(code.to_string())?))
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
            Statement::Let(name, false, Some(sig), val) => {
                format!("let {name}: {} = {}", sig.format(), val.format())
            }
            Statement::Let(name, true, Some(sig), val) => {
                format!("const {name}: {} = {}", sig.format(), val.format())
            }
            Statement::Let(name, false, None, val) => format!("let {name} = {}", val.format()),
            Statement::Let(name, true, None, val) => format!("const {name} = {}", val.format()),
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
    Infix(Box<Operator>),
    List(Vec<Expr>),
    Struct(Vec<(Expr, Expr)>),
    Block(Program),
    Value(Type),
}

impl Expr {
    fn eval(&self, engine: &mut Engine) -> Result<Type, Fault> {
        Ok(match self {
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
        })
    }

    fn parse(source: String) -> Result<Expr, Fault> {
        let token_list: Vec<String> = tokenize(source, SPACE.to_vec())?;
        let token = ok!(token_list.last())?.trim().to_string();
        let token = if let Ok(n) = token.parse::<f64>() {
            Expr::Value(Type::Number(n))
        } else if let Some(sig) = Signature::parse(token.clone()) {
            Expr::Value(Type::Signature(sig))
        // Prefix operators
        } else if token.starts_with("&") {
            let token = token.replacen("&", "", 1);
            Expr::Value(Type::Refer(token))
        } else if token.starts_with("*") {
            let token = token.replacen("*", "", 1);
            Expr::Infix(Box::new(Operator::Derefer(Expr::parse(token)?)))
        } else if token.starts_with("!") {
            let token = token.replacen("!", "", 1);
            Expr::Infix(Box::new(Operator::Not(Expr::parse(token)?)))
        } else if token.starts_with("-") {
            let token = token.replacen("-", "", 1);
            Expr::Infix(Box::new(Operator::Sub(
                Expr::Value(Type::Number(0.0)),
                Expr::parse(token)?,
            )))
        } else if token.starts_with('(') && token.ends_with(')') {
            let token = ok!(token.get(1..token.len() - 1))?.to_string();
            Expr::parse(token)?
        } else if token.starts_with('{') && token.ends_with('}') {
            let token = ok!(token.get(1..token.len() - 1))?.to_string();
            Expr::Block(Engine::parse(token.clone())?)
        } else if token.starts_with("@{") && token.ends_with('}') {
            let token = ok!(token.get(2..token.len() - 1))?.to_string();
            let mut result = Vec::new();
            for i in tokenize(token.clone(), vec![','])? {
                let splited = tokenize(i, vec![':'])?;
                result.push((
                    Expr::parse(ok!(splited.get(0))?.to_string())?,
                    Expr::parse(ok!(splited.get(1))?.to_string())?,
                ));
            }
            Expr::Struct(result)
        } else if token.starts_with('[') && token.ends_with(']') {
            let token = ok!(token.get(1..token.len() - 1))?.to_string();
            let mut list = vec![];
            for elm in tokenize(token, vec![','])? {
                list.push(Expr::parse(elm.trim().to_string())?);
            }
            Expr::List(list)
        } else if token.starts_with('"') && token.ends_with('"') {
            let token = ok!(token.get(1..token.len() - 1))?.to_string();
            Expr::Value(Type::Text(text_escape(token)))
        // Lambda abstract that original formula in the theory
        } else if token.starts_with('λ') && token.contains('.') {
            let token = token.replacen("λ", "", 1);
            let (arg, body) = ok!(token.split_once("."))?;
            Expr::Value(Type::Function(Function::UserDefined(
                arg.to_string(),
                Box::new(Expr::parse(body.to_string())?),
            )))
        // Lambda abstract using back-slash instead of lambda mark
        } else if token.starts_with('\\') && token.contains('.') {
            let token = token.replacen('\\', "", 1);
            let (arg, body) = ok!(token.split_once("."))?;
            Expr::Value(Type::Function(Function::UserDefined(
                arg.to_string(),
                Box::new(Expr::parse(body.to_string())?),
            )))
        // Imperative style syntactic sugar of lambda abstract
        } else if token.starts_with("fn(") && token.contains("->") && token.ends_with(")") {
            let token = token.replacen("fn(", "", 1);
            let token = ok!(token.get(..token.len() - 1))?.to_string();
            let (args, body) = ok!(token.split_once("->"))?;
            let mut args: Vec<&str> = args.split(",").collect();
            args.reverse();
            let mut func = Expr::Value(Type::Function(Function::UserDefined(
                ok!(args.first())?.trim().to_string(),
                Box::new(Expr::parse(body.to_string())?),
            )));
            // Currying
            for arg in ok!(args.get(1..))? {
                func = Expr::Value(Type::Function(Function::UserDefined(
                    arg.trim().to_string(),
                    Box::new(func),
                )));
            }
            func
        // Imperative style syntactic sugar of function application
        } else if token.contains('(') && token.ends_with(')') {
            let token = ok!(token.get(..token.len() - 1))?.to_string();
            let (name, args) = ok!(token.split_once("("))?;
            let args = tokenize(args.to_string(), vec![','])?;
            let mut call = Expr::Infix(Box::new(Operator::Apply(
                Expr::parse(name.to_string())?,
                Expr::parse(ok!(args.first())?.to_string())?,
            )));
            for arg in ok!(args.get(1..))? {
                call = Expr::Infix(Box::new(Operator::Apply(
                    call,
                    Expr::parse(arg.to_string())?,
                )));
            }
            call
        } else if token == "null" {
            Expr::Value(Type::Null)
        } else {
            Expr::Value(Type::Symbol(token))
        };

        if token_list.len() >= 2 {
            let operator = ok!(token_list.get(token_list.len() - 2))?;
            let has_lhs = |len: usize| {
                Expr::parse(
                    ok!(token_list.get(..token_list.len() - len))?.join(&SPACE[0].to_string()),
                )
            };
            Ok(Expr::Infix(Box::new(match operator.as_str() {
                "+" => Operator::Add(has_lhs(2)?, token),
                "-" => Operator::Sub(has_lhs(2)?, token),
                "*" => Operator::Mul(has_lhs(2)?, token),
                "/" => Operator::Div(has_lhs(2)?, token),
                "%" => Operator::Mod(has_lhs(2)?, token),
                "^" => Operator::Pow(has_lhs(2)?, token),
                "==" => Operator::Equal(has_lhs(2)?, token),
                "!=" => Operator::NotEq(has_lhs(2)?, token),
                "<" => Operator::LessThan(has_lhs(2)?, token),
                "<=" => Operator::LessThanEq(has_lhs(2)?, token),
                ">" => Operator::GreaterThan(has_lhs(2)?, token),
                ">=" => Operator::GreaterThanEq(has_lhs(2)?, token),
                "&" => Operator::And(has_lhs(2)?, token),
                "|" => Operator::Or(has_lhs(2)?, token),
                "::" => Operator::Access(has_lhs(2)?, token),
                "as" => Operator::As(has_lhs(2)?, token),
                ":=" => Operator::Assign(has_lhs(2)?, token),
                "|>" => Operator::PipeLine(has_lhs(2)?, token),
                "+=" => Operator::AssignAdd(has_lhs(2)?, token),
                "-=" => Operator::AssignSub(has_lhs(2)?, token),
                "*=" => Operator::AssignMul(has_lhs(2)?, token),
                "/=" => Operator::AssignDiv(has_lhs(2)?, token),
                "%=" => Operator::AssignMod(has_lhs(2)?, token),
                "^=" => Operator::AssignPow(has_lhs(2)?, token),
                operator => {
                    if operator.starts_with("`") && operator.ends_with("`") {
                        let operator = operator[1..operator.len() - 1].to_string();
                        Operator::Apply(
                            Expr::Infix(Box::new(Operator::Apply(
                                Expr::parse(operator)?,
                                has_lhs(2)?,
                            ))),
                            token,
                        )
                    } else {
                        Operator::Apply(has_lhs(1)?, token)
                    }
                }
            })))
        } else {
            Ok(token)
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
        }
    }

    /// Beta reduction of constant arguments when apply function
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
                Operator::Derefer(val) => Operator::Derefer(val.replace(from, to)),
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
                Operator::AssignAdd(lhs, rhs) => {
                    Operator::AssignAdd(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::AssignSub(lhs, rhs) => {
                    Operator::AssignSub(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::AssignMul(lhs, rhs) => {
                    Operator::AssignMul(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::AssignDiv(lhs, rhs) => {
                    Operator::AssignDiv(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::AssignMod(lhs, rhs) => {
                    Operator::AssignMod(lhs.replace(from, to), rhs.replace(from, to))
                }
                Operator::AssignPow(lhs, rhs) => {
                    Operator::AssignPow(lhs.replace(from, to), rhs.replace(from, to))
                }
            })),
            Expr::Block(block) => Expr::Block(
                block
                    .iter()
                    .map(|i| match i {
                        Statement::Print(vals) => {
                            Statement::Print(vals.iter().map(|j| j.replace(from, to)).collect())
                        }
                        Statement::Let(name, protect, sig, val) => Statement::Let(
                            name.clone(),
                            protect.clone(),
                            sig.clone(),
                            val.replace(from, to),
                        ),
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
                    if from.format() == "self" || from.format() == *arg {
                        func.clone()
                    } else {
                        Box::new(func.replace(from, to))
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
        }
    }
}

#[derive(Clone, Debug)]
enum Function {
    BuiltIn(fn(Type, &mut Engine) -> Result<Type, Fault>),
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
    Derefer(Expr),
    As(Expr, Expr),
    Apply(Expr, Expr),
    Assign(Expr, Expr),
    PipeLine(Expr, Expr),
    AssignAdd(Expr, Expr),
    AssignSub(Expr, Expr),
    AssignMul(Expr, Expr),
    AssignDiv(Expr, Expr),
    AssignMod(Expr, Expr),
    AssignPow(Expr, Expr),
}

impl Operator {
    fn eval(&self, engine: &mut Engine) -> Result<Type, Fault> {
        Ok(match self {
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
                    return Err(Fault::Infix(self.clone()));
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
                    let first_index = ok!(lhs.windows(rhs.len()).position(|i| {
                        i.iter().map(|j| j.get_symbol()).collect::<Vec<_>>()
                            == rhs.iter().map(|j| j.get_symbol()).collect::<Vec<_>>()
                    }))?;
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
                    return Err(Fault::Infix(self.clone()));
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
                    return Err(Fault::Infix(self.clone()));
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
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                if rhs.is_match(&lhs) {
                    rhs
                } else {
                    return Err(Fault::Logic);
                }
            }
            Operator::NotEq(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                if !rhs.is_match(&lhs) {
                    rhs
                } else {
                    return Err(Fault::Logic);
                }
            }
            Operator::LessThan(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number()? < rhs.get_number()? {
                    rhs
                } else {
                    return Err(Fault::Logic);
                }
            }
            Operator::LessThanEq(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number()? <= rhs.get_number()? {
                    rhs
                } else {
                    return Err(Fault::Logic);
                }
            }
            Operator::GreaterThan(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number()? > rhs.get_number()? {
                    rhs
                } else {
                    return Err(Fault::Logic);
                }
            }
            Operator::GreaterThanEq(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number()? >= rhs.get_number()? {
                    rhs
                } else {
                    return Err(Fault::Logic);
                }
            }
            Operator::And(lhs, rhs) => {
                let rhs = rhs.eval(engine);
                if lhs.eval(engine).is_ok() && rhs.is_ok() {
                    rhs?
                } else {
                    return Err(Fault::Logic);
                }
            }
            Operator::Or(lhs, rhs) => {
                let lhs = lhs.eval(engine);
                let rhs = rhs.eval(engine);
                if lhs.is_ok() || rhs.is_ok() {
                    rhs.unwrap_or(lhs?)
                } else {
                    return Err(Fault::Logic);
                }
            }
            Operator::Not(val) => {
                let val = val.eval(engine);
                if val.is_ok() {
                    return Err(Fault::Logic);
                } else {
                    Type::Null
                }
            }
            Operator::Access(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                if let (Type::List(list), Type::Number(index)) = (lhs.clone(), rhs.clone()) {
                    ok!(list.get(index as usize))?.clone()
                } else if let (Type::Text(text), Type::Number(index)) = (lhs.clone(), rhs.clone()) {
                    Type::Text(
                        ok!(text.chars().collect::<Vec<char>>().get(index as usize))?.to_string(),
                    )
                } else if let (Type::Struct(st), Type::Text(index)) = (lhs.clone(), rhs.clone()) {
                    ok!(st.get(&index))?.clone()
                } else {
                    return Err(Fault::Infix(self.clone()));
                }
            }
            Operator::Derefer(pointer) => match pointer.clone().eval(engine)? {
                Type::Refer(to) => Expr::Value(Type::Symbol(to.to_string())).eval(engine)?,
                _ => return Err(Fault::Infix(self.clone())),
            },
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
                    Signature::Signature => {
                        Type::Signature(ok!(Signature::parse(lhs.get_text()?))?)
                    }
                }
            }
            Operator::Apply(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                match lhs.get_function()? {
                    Function::BuiltIn(func) => func(rhs, engine)?,
                    Function::UserDefined(parameter, code) => {
                        let code = code
                            .replace(&Expr::Value(Type::Symbol(parameter)), &Expr::Value(rhs))
                            .replace(
                                &Expr::Value(Type::Symbol("self".to_string())),
                                &Expr::Value(lhs),
                            );
                        code.eval(&mut engine.clone())?
                    }
                }
            }
            Operator::Assign(lhs, rhs) => {
                let Expr::Value(Type::Symbol(name)) = lhs else {
                    return Err(Fault::Infix(self.clone()));
                };
                Statement::Let(name.to_owned(), false, None, rhs.to_owned()).eval(engine)?
            }
            Operator::PipeLine(lhs, rhs) => {
                Operator::Apply(rhs.to_owned(), lhs.to_owned()).eval(engine)?
            }
            Operator::AssignAdd(name, rhs) => {
                let lhs = name.eval(engine)?;
                Operator::Assign(
                    name.to_owned(),
                    Expr::Infix(Box::new(Operator::Add(Expr::Value(lhs), rhs.clone()))),
                )
                .eval(engine)?
            }
            Operator::AssignSub(name, rhs) => {
                let lhs = name.eval(engine)?;
                Operator::Assign(
                    name.to_owned(),
                    Expr::Infix(Box::new(Operator::Sub(Expr::Value(lhs), rhs.clone()))),
                )
                .eval(engine)?
            }
            Operator::AssignMul(name, rhs) => {
                let lhs = name.eval(engine)?;
                Operator::Assign(
                    name.to_owned(),
                    Expr::Infix(Box::new(Operator::Mul(Expr::Value(lhs), rhs.clone()))),
                )
                .eval(engine)?
            }
            Operator::AssignDiv(name, rhs) => {
                let lhs = name.eval(engine)?;
                Operator::Assign(
                    name.to_owned(),
                    Expr::Infix(Box::new(Operator::Div(Expr::Value(lhs), rhs.clone()))),
                )
                .eval(engine)?
            }
            Operator::AssignMod(name, rhs) => {
                let lhs = name.eval(engine)?;
                Operator::Assign(
                    name.to_owned(),
                    Expr::Infix(Box::new(Operator::Mod(Expr::Value(lhs), rhs.clone()))),
                )
                .eval(engine)?
            }
            Operator::AssignPow(name, rhs) => {
                let lhs = name.eval(engine)?;
                Operator::Assign(
                    name.to_owned(),
                    Expr::Infix(Box::new(Operator::Pow(Expr::Value(lhs), rhs.clone()))),
                )
                .eval(engine)?
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
            Operator::GreaterThanEq(lhs, rhs) => format!("{} >= {}", lhs.format(), rhs.format()),
            Operator::And(lhs, rhs) => format!("{} & {}", lhs.format(), rhs.format()),
            Operator::Or(lhs, rhs) => format!("{} | {}", lhs.format(), rhs.format()),
            Operator::Not(val) => format!("!{}", val.format()),
            Operator::Access(lhs, rhs) => format!("{} :: {}", lhs.format(), rhs.format()),
            Operator::Derefer(to) => format!("*{}", to.format()),
            Operator::As(lhs, rhs) => format!("{} as {}", lhs.format(), rhs.format()),
            Operator::Assign(lhs, rhs) => format!("{} := {}", lhs.format(), rhs.format()),
            Operator::PipeLine(lhs, rhs) => format!("{} |> {}", lhs.format(), rhs.format()),
            Operator::Apply(lhs, rhs) => format!("{} {}", lhs.format(), rhs.format()),
            Operator::AssignAdd(lhs, rhs) => format!("{} += {}", lhs.format(), rhs.format()),
            Operator::AssignSub(lhs, rhs) => format!("{} -= {}", lhs.format(), rhs.format()),
            Operator::AssignMul(lhs, rhs) => format!("{} *= {}", lhs.format(), rhs.format()),
            Operator::AssignDiv(lhs, rhs) => format!("{} /= {}", lhs.format(), rhs.format()),
            Operator::AssignMod(lhs, rhs) => format!("{} %= {}", lhs.format(), rhs.format()),
            Operator::AssignPow(lhs, rhs) => format!("{} ^= {}", lhs.format(), rhs.format()),
        }
        .to_string()
    }
}

#[derive(Debug, Error)]
enum Fault {
    #[error("access is denied because it's protected memory area")]
    AccessDenied,

    #[error("can not type cast `{}` to {}", value.get_symbol(), to.format())]
    Cast { value: Type, to: Signature },

    #[error("at the IO processing")]
    IO,

    #[error("the result value `{}` is different to expected type `{}`", value.get_symbol(), annotate.format())]
    Type { value: Type, annotate: Signature },

    #[error("missmatching of arguments length when function application")]
    MissMatchArgLen,

    #[error("the program is not able to parse. check out is the syntax correct")]
    Syntax,

    #[error("can not evaluate infix `{}`", _0.format())]
    Infix(Operator),

    #[error("the logical operation has bankruptcy")]
    Logic,
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
            Type::Text(text) => format!(
                "\"{}\"",
                text.replace("\\", "\\\\")
                    .replace("'", "\\'")
                    .replace("\"", "\\\"")
                    .replace("`", "\\`")
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                    .replace("\r", "\\r")
            ),
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

    fn get_number(&self) -> Result<f64, Fault> {
        let err = Err(Fault::Cast {
            value: self.clone(),
            to: Signature::Number,
        });
        match self {
            Type::Number(n) => Ok(n.to_owned()),
            Type::Symbol(s) | Type::Text(s) => {
                if let Ok(n) = s.trim().parse::<f64>() {
                    Ok(n)
                } else {
                    err
                }
            }
            Type::Null => Ok(0.0),
            _ => err,
        }
    }

    fn get_text(&self) -> Result<String, Fault> {
        match self {
            Type::Symbol(s) | Type::Text(s) => Ok(s.to_string()),
            Type::Number(n) => Ok(n.to_string()),
            Type::Signature(s) => Ok(s.format()),
            Type::Null => Ok(String::new()),
            _ => Err(Fault::Cast {
                value: self.clone(),
                to: Signature::Text,
            }),
        }
    }

    fn get_list(&self) -> Vec<Type> {
        match self {
            Type::List(list) => list.to_owned(),
            Type::Text(text) => text.chars().map(|i| Type::Text(i.to_string())).collect(),
            Type::Null => Vec::new(),
            other => vec![other.to_owned()],
        }
    }

    fn get_function(&self) -> Result<Function, Fault> {
        if let Type::Function(func) = self {
            Ok(func.to_owned())
        } else {
            Err(Fault::Cast {
                value: self.clone(),
                to: Signature::Function,
            })
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

    fn get_signature(&self) -> Result<Signature, Fault> {
        if let Type::Signature(sig) = self {
            Ok(sig.clone())
        } else {
            Err(Fault::Cast {
                value: self.clone(),
                to: Signature::Signature,
            })
        }
    }

    fn get_struct(&self) -> Result<HashMap<String, Type>, Fault> {
        if let Type::Struct(val) = self {
            Ok(val.clone())
        } else {
            Err(Fault::Cast {
                value: self.clone(),
                to: Signature::Struct,
            })
        }
    }

    fn get_refer(&self) -> Result<String, Fault> {
        if let Type::Refer(val) = self {
            Ok(val.clone())
        } else {
            Err(Fault::Cast {
                value: self.clone(),
                to: Signature::Refer,
            })
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

fn tokenize(input: String, delimiter: Vec<char>) -> Result<Vec<String>, Fault> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current_token = String::new();
    let mut in_parentheses: usize = 0;
    let mut in_quote = false;
    let mut is_escape = false;

    for c in input.chars() {
        if is_escape {
            current_token.push(match c {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                _ => c,
            });
            is_escape = false;
        } else {
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
                        return Err(Fault::Syntax);
                    }
                }
                '"' | '\'' | '`' => {
                    in_quote = !in_quote;
                    current_token.push(c);
                }
                '\\' if in_quote => {
                    current_token.push(c);
                    is_escape = true;
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
    }

    // Syntax error check
    if is_escape || in_quote || in_parentheses != 0 {
        return Err(Fault::Syntax);
    }
    if !current_token.is_empty() {
        tokens.push(current_token.clone());
        current_token.clear();
    }
    Ok(tokens)
}

fn text_escape(text: String) -> String {
    let mut result = String::new();
    let mut is_escape = false;
    for c in text.chars() {
        if is_escape {
            result.push(c);
            is_escape = false;
        } else {
            match c {
                '\\' => {
                    is_escape = true;
                }
                _ => result.push(c),
            }
        }
    }
    result
}
