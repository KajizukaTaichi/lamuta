use clap::Parser;
use colored::*;
use indexmap::IndexMap;
use reqwest::blocking;
use rustyline::{error::ReadlineError, DefaultEditor};
use std::{
    fs::{read_to_string, File},
    io::{self, Write},
    process::exit,
    thread::sleep,
    time::Duration,
};
use thiserror::Error;

const VERSION: &str = "0.4.2";
const SPACE: [char; 5] = [' ', '　', '\n', '\t', '\r'];
const BUILTIN: [&str; 14] = [
    "type",
    "env",
    "free",
    "eval",
    "alphaConvert",
    "input",
    "readFile",
    "range",
    "load",
    "save",
    "std",
    "sleep",
    "exit",
    "cmdLineArgs",
];

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

macro_rules! some {
    ($result_value: expr) => {
        if let Ok(ok) = $result_value {
            Some(ok)
        } else {
            None
        }
    };
}

#[derive(Parser)]
#[command(
    name = "Lamuta", version = VERSION,
    about = "A functional programming language that can write lambda calculus formula as they are"
)]
struct Cli {
    /// Source file to evaluate
    #[arg(index = 1)]
    file: Option<String>,

    /// Command-line arguments to pass the script
    #[arg(index = 2, value_name = "ARGS", num_args = 0..)]
    args_position: Option<Vec<String>>,

    /// Optional command-line arguments
    #[arg(short = 'a', long = "args", value_name = "ARGS", num_args = 0..)]
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
            false,
            Expr::Value(Type::Text(file)),
        )
        .eval(&mut engine)
        {
            eprintln!("{}: {e}", "Fault".red())
        }
    } else {
        println!("{title} {VERSION}", title = "Lamuta".blue().bold());
        let mut rl = DefaultEditor::new().unwrap();
        let mut session = 1;

        macro_rules! repl_print {
            ($color: ident, $value: expr) => {
                println!("{navi} {}", $value, navi = "=>".$color())
            };
        }
        macro_rules! fault {
            ($e: expr) => {
                repl_print!(red, format!("Fault: {}", $e))
            };
        }

        loop {
            match rl.readline(&format!("[{session:0>3}]> ")) {
                Ok(code) => {
                    match Engine::parse(code) {
                        Ok(ast) => match engine.eval(ast) {
                            Ok(result) => repl_print!(green, result.format()),
                            Err(e) => fault!(e),
                        },
                        Err(e) => fault!(e),
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

type Scope = IndexMap<String, Type>;
type Program = Vec<Statement>;
#[derive(Debug, Clone)]
struct Engine {
    env: Scope,
    protect: Vec<String>,
}

impl Engine {
    fn new() -> Engine {
        Engine {
            protect: BUILTIN.to_vec().iter().map(|i| i.to_string()).collect(),
            env: IndexMap::from([
                (
                    "type".to_string(),
                    Type::Function(Function::BuiltIn(|expr, _| {
                        Ok(Type::Signature(expr.type_of()))
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
                        ok!(
                            engine.env.shift_remove(name),
                            Fault::Refer(name.to_string())
                        )?;
                        Ok(Type::Null)
                    })),
                ),
                (
                    "eval".to_string(),
                    Type::Function(Function::BuiltIn(|args, engine| {
                        let args = args.get_list()?;
                        let code = ok!(args.get(0), Fault::ArgLen)?.get_text()?;
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
                        let args = args.get_list()?;
                        let func = ok!(args.get(0), Fault::ArgLen)?;
                        let new_name = ok!(args.get(1), Fault::ArgLen)?.get_text()?;
                        let Type::Function(Function::UserDefined(old_name, body)) = func else {
                            return Err(Fault::Type(func.to_owned(), Signature::Function));
                        };
                        Ok(Type::Function(Function::UserDefined(
                            new_name.clone(),
                            Box::new(body.replace(
                                &Expr::Value(Type::Symbol(old_name.to_owned())),
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
                    "readFile".to_string(),
                    Type::Function(Function::BuiltIn(|i, _| {
                        Ok(Type::Text(ok!(
                            some!(read_to_string(i.get_text()?)),
                            Fault::IO
                        )?))
                    })),
                ),
                (
                    "range".to_string(),
                    Type::Function(Function::BuiltIn(|params, _| {
                        let params = params.get_list()?;
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
                            let start = params[0].get_number()?;
                            let end = params[1].get_number()?;
                            let is_positive = (end - start).is_sign_positive();
                            while if is_positive {
                                current < params[1].get_number()?
                            } else {
                                current > params[1].get_number()?
                            } {
                                range.push(Type::Number(current));
                                current += if is_positive { 1.0 } else { -1.0 };
                            }
                            Ok(Type::List(range))
                        } else if params.len() == 3 {
                            let mut range: Vec<Type> = vec![];
                            let mut current: f64 = params[0].get_number()?;
                            let step = params[2].get_number()?;
                            if step == 0.0 {
                                return Err(Fault::Logic(Operator::Apply(
                                    Expr::Value(Type::Symbol("range".to_string())),
                                    false,
                                    Expr::Value(Type::List(params)),
                                )));
                            }
                            while if step.is_sign_positive() {
                                current < params[1].get_number()?
                            } else {
                                current > params[1].get_number()?
                            } {
                                range.push(Type::Number(current));
                                current += step;
                            }
                            Ok(Type::List(range))
                        } else {
                            Err(Fault::ArgLen)
                        }
                    })),
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
                            if !BUILTIN.contains(&k.as_str()) {
                                render += &format!("let {k} = {};\n", v.format());
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
                (
                    "std".to_string(),
                    Type::Text("https://kajizukataichi.github.io/lamuta/lib/std.lm".to_string()),
                ),
                (
                    "sleep".to_string(),
                    Type::Function(Function::BuiltIn(|i, _| {
                        sleep(Duration::from_secs_f64(i.get_number()?));
                        Ok(Type::Null)
                    })),
                ),
                (
                    "exit".to_string(),
                    Type::Function(Function::BuiltIn(|arg, _| exit(arg.get_number()? as i32))),
                ),
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
    Let(Expr, bool, Option<Signature>, Expr),
    If(Expr, Expr, Option<Expr>),
    Match(Expr, Vec<(Vec<Expr>, Expr)>),
    For(Expr, Expr, Expr),
    While(Expr, Expr),
    Fault(Option<Expr>),
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
                            other => other.format(),
                        }
                    );
                }
                io::stdout().flush().unwrap();
                Type::Null
            }
            Statement::Let(name, protect, sig, expr) => {
                let val = expr.eval(engine)?;
                if let Some(sig) = sig {
                    if val.type_of().format() != sig.format() {
                        return Err(Fault::Type(val, sig.to_owned()));
                    }
                }
                if let Expr::Value(Type::Symbol(name)) = name {
                    if engine.protect.contains(name) {
                        return Err(Fault::AccessDenied);
                    }
                    if name != "_" {
                        engine.env.insert(name.to_owned(), val.clone());
                        if *protect {
                            engine.protect.push(name.to_owned());
                        }
                    }
                } else if let Expr::List(list) = name {
                    let val = val.get_list()?;
                    if list.len() == val.len() {
                        for (name, val) in list.iter().zip(val) {
                            Statement::Let(name.to_owned(), false, None, Expr::Value(val))
                                .eval(engine)?;
                        }
                    } else {
                        return Err(Fault::Syntax);
                    }
                } else if let Expr::Infix(infix) = name {
                    let infix = *infix.clone();
                    if let Operator::Access(accessor, key) = infix {
                        let obj = accessor.eval(engine)?;
                        let key = key.eval(engine)?;
                        let updated_obj = obj.modify_inside(key, val.clone(), engine)?;
                        Statement::Let(accessor, false, None, Expr::Value(updated_obj.clone()))
                            .eval(engine)?;
                    } else if let Operator::Derefer(pointer) = infix {
                        let pointer = pointer.eval(engine)?.get_refer()?;
                        Statement::Let(
                            Expr::Value(Type::Symbol(pointer)),
                            false,
                            None,
                            Expr::Value(val.clone()),
                        )
                        .eval(engine)?;
                    } else {
                        return Err(Fault::Syntax);
                    }
                } else {
                    return Err(Fault::Syntax);
                }
                val
            }
            Statement::If(expr, then, r#else) => match expr.eval(engine) {
                Ok(it) => {
                    Statement::Let(
                        Expr::Value(Type::Symbol("it".to_string())),
                        false,
                        None,
                        Expr::Value(it),
                    )
                    .eval(engine)?;
                    then.eval(engine)?
                }
                Err(err) => {
                    if let Some(r#else) = r#else {
                        Statement::Let(
                            Expr::Value(Type::Symbol("err".to_string())),
                            false,
                            None,
                            Expr::Value(Type::Text(format!("{err}"))),
                        )
                        .eval(engine)?;
                        r#else.eval(engine)?
                    } else {
                        Type::Null
                    }
                }
            },
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
                for i in expr.eval(engine)?.get_list()? {
                    Statement::Let(counter.clone(), false, None, Expr::Value(i)).eval(engine)?;
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
            Statement::Fault(Some(msg)) => {
                return Err(Fault::General(Some(msg.eval(engine)?.get_text()?)))
            }
            Statement::Fault(None) => return Err(Fault::General(None)),
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
            let splited = tokenize(code.to_string(), vec!['='])?;
            let (name, code) = (ok!(splited.get(0))?, ok!(splited.get(1))?);
            let splited = tokenize(name.to_string(), vec![':'])?;
            if let (Some(name), Some(sig)) = (splited.get(0), splited.get(1)) {
                Ok(Statement::Let(
                    Expr::parse(name.trim().to_string())?,
                    false,
                    Some(ok!(Signature::parse(sig.trim().to_string()))?),
                    Expr::parse(code.to_string())?,
                ))
            } else {
                Ok(Statement::Let(
                    Expr::parse(name.trim().to_string())?,
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
                Expr::parse(name.trim().to_string())?,
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
                    Some(Expr::Block(Engine::parse(
                        ok!(code.get(pos + 1..))?.join(&SPACE[0].to_string()),
                    )?)),
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
                    Expr::parse(ok!(code.get(0))?.to_string())?,
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
        } else if code.starts_with("fault") {
            let code = code["fault".len()..].to_string();
            Ok(Statement::Fault(some!(Expr::parse(code.to_string()))))
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
                format!("let {}: {} = {}", name.format(), sig.format(), val.format())
            }
            Statement::Let(name, true, Some(sig), val) => {
                format!(
                    "const {}: {} = {}",
                    name.format(),
                    sig.format(),
                    val.format()
                )
            }
            Statement::Let(name, false, None, val) => {
                format!("let {} = {}", name.format(), val.format())
            }
            Statement::Let(name, true, None, val) => {
                format!("const {} = {}", name.format(), val.format())
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
                format!(
                    "for {} in {} {}",
                    counter.format(),
                    iterator.format(),
                    code.format()
                )
            }
            Statement::While(cond, code) => {
                format!("while {} {}", cond.format(), code.format())
            }
            Statement::Fault(Some(msg)) => format!("fault {}", msg.format()),
            Statement::Fault(None) => "fault".to_string(),
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
                let mut result = IndexMap::new();
                for (k, x) in st {
                    result.insert(k.eval(engine)?.get_text()?, x.eval(engine)?);
                }
                Type::Struct(result)
            }
            Expr::Value(Type::Symbol(name)) => {
                if name != "_" {
                    if let Some(refer) = engine.env.get(name.as_str()) {
                        refer.clone()
                    } else {
                        return Err(Fault::Refer(name.to_string()));
                    }
                } else {
                    Type::Symbol(name.to_string())
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
            Expr::Block(Engine::parse(token)?)
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
        } else if (token.starts_with('"') && token.ends_with('"'))
            || (token.starts_with("'") && token.ends_with("'"))
        {
            let text = ok!(token.get(1..token.len() - 1))?.to_string();
            Expr::Value(Type::Text(text_escape(text)))
        // Functionize operator
        } else if token.starts_with("`") && token.ends_with("`") {
            let token = ok!(token.get(1..token.len() - 1))?.trim().to_string();
            let source = format!("λx.λy.(x {token} y)");
            let expr = Expr::parse(source.clone())?;
            if expr.format() != source {
                return Err(Fault::Syntax);
            }
            expr
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
                Box::new(Expr::Block(Engine::parse(body.to_string())?)),
            )));
            // Currying
            for arg in ok!(args.get(1..))? {
                func = Expr::Value(Type::Function(Function::UserDefined(
                    arg.trim().to_string(),
                    Box::new(func),
                )));
            }
            func
        // Object-oriented style syntactic sugar of access operator
        } else if tokenize(token.clone(), vec!['.'])?.len() >= 2 {
            let args = tokenize(token, vec!['.'])?;
            Expr::Infix(Box::new(Operator::Access(
                Expr::parse(ok!(args.get(0..args.len() - 1))?.join("."))?,
                Expr::Value(Type::Text(ok!(args.last())?.trim().to_string())),
            )))
        // Imperative style syntactic sugar of list access by index
        } else if token.contains('[') && token.ends_with(']') {
            let token = ok!(token.get(..token.len() - 1))?.to_string();
            let (name, args) = ok!(token.split_once("["))?;
            Expr::Infix(Box::new(Operator::Access(
                Expr::Value(Type::Symbol(name.trim().to_string())),
                Expr::parse(args.to_string())?,
            )))
        // Imperative style syntactic sugar of function application
        } else if token.contains('(') && token.ends_with(')') {
            let token = ok!(token.get(..token.len() - 1))?.to_string();
            let (name, args) = ok!(token.split_once("("))?;
            let is_lazy = name.ends_with("?");
            let args = tokenize(args.to_string(), vec![','])?;
            let mut call = Expr::Infix(Box::new(Operator::Apply(
                Expr::Value(Type::Symbol(
                    if is_lazy {
                        ok!(name.get(..name.len() - 1))?
                    } else {
                        name
                    }
                    .trim()
                    .to_string(),
                )),
                is_lazy,
                Expr::parse(ok!(args.first())?.to_string())?,
            )));
            for arg in ok!(args.get(1..))? {
                call = Expr::Infix(Box::new(Operator::Apply(
                    call,
                    is_lazy,
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
                "?" => Operator::Apply(has_lhs(2)?, true, token),
                "::" => Operator::Access(has_lhs(2)?, token),
                "as" => Operator::As(has_lhs(2)?, token),
                "|>" => Operator::PipeLine(has_lhs(2)?, token),
                ":=" => Operator::Assign(has_lhs(2)?, token),
                "+=" => Operator::AssignAdd(has_lhs(2)?, token),
                "-=" => Operator::AssignSub(has_lhs(2)?, token),
                "*=" => Operator::AssignMul(has_lhs(2)?, token),
                "/=" => Operator::AssignDiv(has_lhs(2)?, token),
                "%=" => Operator::AssignMod(has_lhs(2)?, token),
                "^=" => Operator::AssignPow(has_lhs(2)?, token),
                "~" => Operator::To(has_lhs(2)?, token),
                operator => {
                    if operator.starts_with("`") && operator.ends_with("`") {
                        let operator = operator[1..operator.len() - 1].to_string();
                        Operator::Apply(
                            Expr::Infix(Box::new(Operator::Apply(
                                Expr::parse(operator)?,
                                false,
                                has_lhs(2)?,
                            ))),
                            false,
                            token,
                        )
                    } else {
                        Operator::Apply(has_lhs(1)?, false, token)
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
            Expr::Value(val) => val.format(),
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
                Operator::Apply(lhs, is_lazy, rhs) => {
                    Operator::Apply(lhs.replace(from, to), is_lazy, rhs.replace(from, to))
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
                Operator::To(lhs, rhs) => {
                    Operator::To(lhs.replace(from, to), rhs.replace(from, to))
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
                        Statement::Fault(Some(msg)) => {
                            Statement::Fault(Some(msg.replace(from, to)))
                        }
                        Statement::Fault(None) => Statement::Fault(None),
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
    Apply(Expr, bool, Expr),
    PipeLine(Expr, Expr),
    Assign(Expr, Expr),
    AssignAdd(Expr, Expr),
    AssignSub(Expr, Expr),
    AssignMul(Expr, Expr),
    AssignDiv(Expr, Expr),
    AssignMod(Expr, Expr),
    AssignPow(Expr, Expr),
    To(Expr, Expr),
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
                } else if let (Type::List(mut list1), Type::List(list2)) =
                    (lhs.clone(), &rhs.clone())
                {
                    let first_index = ok!(
                        list1.windows(list2.len()).position(|i| {
                            i.iter().map(|j| j.format()).collect::<Vec<_>>()
                                == list2.iter().map(|j| j.format()).collect::<Vec<_>>()
                        }),
                        Fault::Index(rhs, lhs)
                    )?;
                    for _ in 0..list2.len() {
                        list1.remove(first_index);
                    }
                    Type::List(list1)
                } else if let (Type::Struct(mut st), Type::Text(key)) = (lhs.clone(), &rhs) {
                    ok!(st.shift_remove(key), Fault::Key(rhs, lhs))?;
                    Type::Struct(st)
                } else if let (Type::List(mut list), Type::Number(index)) = (lhs.clone(), &rhs) {
                    if 0.0 <= *index && *index < list.len() as f64 {
                        list.remove(index.clone() as usize);
                        Type::List(list)
                    } else {
                        return Err(Fault::Index(rhs, lhs));
                    }
                } else if let (Type::Text(text), Type::Number(index)) = (&lhs, &rhs) {
                    if 0.0 <= *index && *index < text.chars().count() as f64 {
                        let mut chars: Vec<char> = text.chars().collect();
                        chars.remove(index.clone() as usize);
                        Type::Text(
                            chars
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<String>>()
                                .concat(),
                        )
                    } else {
                        return Err(Fault::Index(rhs, lhs));
                    }
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
                    return Err(Fault::Logic(self.clone()));
                }
            }
            Operator::NotEq(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                if !rhs.is_match(&lhs) {
                    rhs
                } else {
                    return Err(Fault::Logic(self.clone()));
                }
            }
            Operator::LessThan(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number()? < rhs.get_number()? {
                    rhs
                } else {
                    return Err(Fault::Logic(self.clone()));
                }
            }
            Operator::LessThanEq(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number()? <= rhs.get_number()? {
                    rhs
                } else {
                    return Err(Fault::Logic(self.clone()));
                }
            }
            Operator::GreaterThan(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number()? > rhs.get_number()? {
                    rhs
                } else {
                    return Err(Fault::Logic(self.clone()));
                }
            }
            Operator::GreaterThanEq(lhs, rhs) => {
                let rhs = rhs.eval(engine)?;
                if lhs.eval(engine)?.get_number()? >= rhs.get_number()? {
                    rhs
                } else {
                    return Err(Fault::Logic(self.clone()));
                }
            }
            Operator::And(lhs, rhs) => {
                let rhs = rhs.eval(engine);
                if lhs.eval(engine).is_ok() && rhs.is_ok() {
                    rhs?
                } else {
                    return Err(Fault::Logic(self.clone()));
                }
            }
            Operator::Or(lhs, rhs) => {
                let lhs = lhs.eval(engine);
                let rhs = rhs.eval(engine);
                if lhs.is_ok() || rhs.is_ok() {
                    rhs.unwrap_or(lhs?)
                } else {
                    return Err(Fault::Logic(self.clone()));
                }
            }
            Operator::Not(val) => {
                let val = val.eval(engine);
                if val.is_ok() {
                    return Err(Fault::Logic(self.clone()));
                } else {
                    Type::Null
                }
            }
            Operator::Access(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                if let (Type::List(list), Type::Number(index)) = (lhs.clone(), rhs.clone()) {
                    ok!(list.get(index as usize), Fault::Index(rhs, lhs))?.clone()
                } else if let (Type::Text(text), Type::Number(index)) = (lhs.clone(), rhs.clone()) {
                    Type::Text(
                        ok!(
                            text.chars().collect::<Vec<char>>().get(index as usize),
                            Fault::Index(rhs.clone(), lhs.clone())
                        )?
                        .to_string(),
                    )
                } else if let (Type::Struct(st), Type::Text(index)) = (lhs.clone(), rhs.clone()) {
                    ok!(st.get(&index), Fault::Key(rhs, lhs))?.clone()
                } else if let (Type::List(list), Type::List(index)) = (lhs.clone(), rhs.clone()) {
                    let mut result = vec![];
                    for i in index {
                        result.push(ok!(
                            list.get(i.get_number()? as usize).cloned(),
                            Fault::Index(rhs.clone(), lhs.clone())
                        )?);
                    }
                    Type::List(result)
                } else if let (Type::Text(text), Type::List(index)) = (lhs.clone(), rhs.clone()) {
                    let mut result = String::new();
                    for i in index {
                        result.push(ok!(
                            text.chars()
                                .collect::<Vec<char>>()
                                .get(i.get_number()? as usize)
                                .cloned(),
                            Fault::Index(rhs.clone(), lhs.clone())
                        )?);
                    }
                    Type::Text(result)
                } else {
                    return Err(Fault::Infix(self.clone()));
                }
            }
            Operator::Derefer(pointer) => {
                let to = pointer.clone().eval(engine)?.get_refer()?;
                Expr::Value(Type::Symbol(to.to_string())).eval(engine)?
            }
            Operator::As(lhs, rhs) => {
                let lhs = lhs.eval(engine)?;
                let rhs = rhs.eval(engine)?;
                lhs.cast(rhs.get_signature()?)?
            }
            Operator::Apply(lhs, is_lazy, rhs) => {
                let lhs = lhs.eval(engine)?;
                if let Type::Function(func) = lhs.clone() {
                    match func {
                        Function::BuiltIn(func) => func(rhs.eval(engine)?, engine)?,
                        Function::UserDefined(parameter, code) => {
                            let code = code
                                .replace(
                                    &Expr::Value(Type::Symbol(parameter)),
                                    &if *is_lazy {
                                        rhs.clone()
                                    } else {
                                        Expr::Value(rhs.eval(engine)?)
                                    },
                                )
                                .replace(
                                    &Expr::Value(Type::Symbol("self".to_string())),
                                    &Expr::Value(lhs),
                                );
                            code.eval(&mut engine.clone())?
                        }
                    }
                } else {
                    return Err(Fault::Apply(lhs));
                }
            }
            Operator::PipeLine(lhs, rhs) => {
                Operator::Apply(rhs.to_owned(), false, lhs.to_owned()).eval(engine)?
            }
            Operator::Assign(lhs, rhs) => {
                Statement::Let(lhs.to_owned(), false, None, rhs.to_owned()).eval(engine)?
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
            Operator::To(from, to) => Operator::Apply(
                Expr::Value(Type::Symbol("range".to_string())),
                false,
                Expr::List(vec![from.to_owned(), to.to_owned()]),
            )
            .eval(engine)?,
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
            Operator::Apply(lhs, true, rhs) => format!("{} ? {}", lhs.format(), rhs.format()),
            Operator::Apply(lhs, false, rhs) => format!("{} {}", lhs.format(), rhs.format()),
            Operator::AssignAdd(lhs, rhs) => format!("{} += {}", lhs.format(), rhs.format()),
            Operator::AssignSub(lhs, rhs) => format!("{} -= {}", lhs.format(), rhs.format()),
            Operator::AssignMul(lhs, rhs) => format!("{} *= {}", lhs.format(), rhs.format()),
            Operator::AssignDiv(lhs, rhs) => format!("{} /= {}", lhs.format(), rhs.format()),
            Operator::AssignMod(lhs, rhs) => format!("{} %= {}", lhs.format(), rhs.format()),
            Operator::AssignPow(lhs, rhs) => format!("{} ^= {}", lhs.format(), rhs.format()),
            Operator::To(lhs, rhs) => format!("{} ~ {}", lhs.format(), rhs.format()),
        }
        .to_string()
    }
}

#[derive(Debug, Error)]
enum Fault {
    #[error("can not apply function because `{}` is not lambda abstract", _0.format())]
    Apply(Type),

    #[error("key `{}` is not found in the struct `{}`", _0.format(), _1.format())]
    Key(Type, Type),

    #[error("index `{}` is out of the list `{}`", _0.format(), _1.format())]
    Index(Type, Type),

    #[error("access is denied because it's protected memory area")]
    AccessDenied,

    #[error("can not access undefined variable `{0}`")]
    Refer(String),

    #[error("can not type cast `{}` to {}", _0.format(), _1.format())]
    Cast(Type, Signature),

    #[error("at the IO processing has problem")]
    IO,

    #[error("the value `{}` is different to expected type `{}`", _0.format(), _1.format())]
    Type(Type, Signature),

    #[error("missmatching of arguments length when function application")]
    ArgLen,

    #[error("the program is not able to parse. check out is the syntax correct")]
    Syntax,

    #[error("can not evaluate expression `{}`", _0.format())]
    Infix(Operator),

    #[error("the logical operation `{}` has bankruptcy", _0.format())]
    Logic(Operator),

    #[error("{}", if let Some(msg) = _0 { msg } else { "throwed by user-defined program" })]
    General(Option<String>),
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
    Struct(IndexMap<String, Type>),
    Null,
}

impl Type {
    fn cast(&self, sig: Signature) -> Result<Type, Fault> {
        let err = Err(Fault::Cast(self.clone(), sig.clone()));
        Ok(match sig {
            Signature::Number => Type::Number(match self {
                Type::Number(n) => n.to_owned(),
                Type::Symbol(s) | Type::Text(s) => {
                    if let Ok(n) = s.trim().parse::<f64>() {
                        n
                    } else {
                        return err;
                    }
                }
                Type::Null => 0.0,
                _ => return err,
            }),
            Signature::Symbol => Type::Symbol(self.format()),
            Signature::Text => Type::Text(match self {
                Type::Symbol(s) | Type::Text(s) => s.to_string(),
                Type::Number(n) => n.to_string(),
                Type::Signature(s) => s.format(),
                Type::Null => String::new(),
                _ => return err,
            }),
            Signature::List => Type::List(match self {
                Type::List(list) => list.to_owned(),
                Type::Text(text) => text.chars().map(|i| Type::Text(i.to_string())).collect(),
                Type::Struct(strct) => strct
                    .iter()
                    .map(|(k, y)| Type::List(vec![Type::Text(k.to_owned()), y.to_owned()]))
                    .collect(),
                Type::Null => Vec::new(),
                _ => return err,
            }),
            _ => return err,
        })
    }

    fn format(&self) -> String {
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
                l.iter().map(|i| i.format()).collect::<Vec<_>>().join(", ")
            ),
            Type::Signature(sig) => sig.format(),
            Type::Refer(to) => format!("&{to}"),
            Type::Struct(val) => format!(
                "@{{ {} }}",
                val.iter()
                    .map(|(k, v)| format!("\"{k}\": {}", v.format()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    fn get_number(&self) -> Result<f64, Fault> {
        match self {
            Type::Number(n) => Ok(n.to_owned()),
            _ => Err(Fault::Type(self.clone(), Signature::Number)),
        }
    }

    fn get_refer(&self) -> Result<String, Fault> {
        match self {
            Type::Refer(r) => Ok(r.to_owned()),
            _ => Err(Fault::Type(self.clone(), Signature::Refer)),
        }
    }

    fn get_text(&self) -> Result<String, Fault> {
        match self {
            Type::Text(s) => Ok(s.to_string()),
            _ => Err(Fault::Type(self.clone(), Signature::Text)),
        }
    }

    fn get_list(&self) -> Result<Vec<Type>, Fault> {
        match self {
            Type::List(list) => Ok(list.to_owned()),
            _ => Err(Fault::Type(self.clone(), Signature::List)),
        }
    }

    fn get_signature(&self) -> Result<Signature, Fault> {
        match self {
            Type::Signature(sig) => Ok(sig.to_owned()),
            _ => Err(Fault::Type(self.clone(), Signature::Signature)),
        }
    }

    fn get_struct(&self) -> Result<IndexMap<String, Type>, Fault> {
        match self {
            Type::Struct(st) => Ok(st.to_owned()),
            _ => Err(Fault::Type(self.clone(), Signature::Struct)),
        }
    }

    fn type_of(&self) -> Signature {
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
            if pattern.format() == "_" {
                true
            } else {
                self.format() == pattern.format()
            }
        }
    }

    fn modify_inside(&self, index: Type, val: Type, engine: &mut Engine) -> Result<Type, Fault> {
        macro_rules! assign {
            ($list: expr, $index: expr, $proc: block) => {
                if 0.0 <= $index && $index < $list.len() as f64 {
                    $proc
                } else {
                    return Err(Fault::Index(Type::Number($index), self.to_owned()));
                }
            };
        }
        macro_rules! range_check {
            ($first_index: expr, $index: expr) => {
                if Type::List($index.clone()).format()
                    != Operator::To(
                        Expr::Value(Type::Number($first_index)),
                        Expr::Value(Type::Number($first_index + $index.len() as f64)),
                    )
                    .eval(engine)?
                    .format()
                {
                    return Err(Fault::Index(Type::List($index.clone()), self.clone()));
                }
            };
        }
        macro_rules! delete_iter {
            ($list: expr, $first_index: expr, $index: expr) => {
                for _ in 0..$index.len() {
                    assign!($list, $first_index, {
                        $list.remove($first_index as usize);
                    });
                }
            };
        }
        macro_rules! first_index {
            ($index: expr) => {
                ok!(
                    $index.first(),
                    Fault::Index(Type::List($index.clone()), self.to_owned())
                )?
                .get_number()?
            };
        }
        macro_rules! char_vec {
            ($text: expr) => {
                $text
                    .chars()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
            };
        }

        Ok(
            if let (Type::List(mut list), Type::Number(index)) = (self.clone(), index.clone()) {
                assign!(list, index, {
                    list[index as usize] = val.clone();
                    Type::List(list)
                })
            } else if let (Type::Text(text), Type::Number(index)) = (self.clone(), index.clone()) {
                let mut text = char_vec!(text);
                assign!(text, index, {
                    text[index as usize] = val.get_text()?;
                    Type::Text(text.concat())
                })
            } else if let (Type::Struct(mut st), Type::Text(index)) = (self.clone(), index.clone())
            {
                st.insert(index, val.clone());
                Type::Struct(st)
            } else if let (Type::List(mut list), Type::List(index)) = (self.clone(), index.clone())
            {
                let first_index = first_index!(index);
                range_check!(first_index, index);
                delete_iter!(list, first_index, index);
                list.insert(first_index as usize, val.clone());
                Type::List(list)
            } else if let (Type::Text(text), Type::List(index)) = (self.clone(), index.clone()) {
                let mut text = char_vec!(text);
                let first_index = first_index!(index);
                range_check!(first_index, index);
                delete_iter!(text, first_index, index);
                text.insert(first_index as usize, val.get_text()?);
                Type::Text(text.concat())
            } else {
                return Err(Fault::Infix(Operator::Access(
                    Expr::Value(self.clone()),
                    Expr::Value(index),
                )));
            },
        )
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
