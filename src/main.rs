mod utils;
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
use unicode_xid::UnicodeXID;

const VERSION: &str = "0.4.2";
const SPACE: [char; 5] = [' ', '　', '\n', '\t', '\r'];
const RESERVED: [&str; 10] = [
    "print", "let", "const", "if", "else", "match", "for", "in", "while", "fault",
];
const BUILTIN: [&str; 12] = [
    "type", "std", "env", "free", "eval", "alphaConvert",
    "input", "readFile", "load", "save", "sleep", "exit"
];

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
        crash!(engine.alloc(
            &"cmdLineArgs".to_string(),
            &Type::List(args.iter().map(|i| Type::Text(i.to_owned())).collect()),
        ));
    }

    if let Some(file) = cli.file {
        crash!(Operator::Apply(
            Expr::Value(Type::Symbol("load".to_string())),
            false,
            Expr::Value(Type::Text(file)),
        )
        .eval(&mut engine))
    } else {
        println!("{title} {VERSION}", title = "Lamuta".blue().bold());
        let mut rl = DefaultEditor::new().unwrap();
        let mut session = 1;

        loop {
            match rl.readline(&format!("[{session:0>3}]> ")) {
                Ok(code) => {
                    match Engine::parse(&code) {
                        Ok(ast) => match engine.eval(&ast) {
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
                    "std".to_string(),
                    Type::Text("https://kajizukataichi.github.io/lamuta/lib/std.lm".to_string()),
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
                        engine.free(name)?;
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
                            engine.eval(&Engine::parse(&code)?)
                        } else {
                            engine.eval(&Engine::parse(&code)?)
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
                    "load".to_string(),
                    Type::Function(Function::BuiltIn(|expr, engine| {
                        let name = expr.get_text()?;
                        if let Ok(module) = read_to_string(&name) {
                            engine.eval(&Engine::parse(&module)?)
                        } else if let Ok(module) = blocking::get(name) {
                            if let Ok(code) = module.text() {
                                engine.eval(&Engine::parse(&code)?)
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
                            if BUILTIN.contains(&k.as_str()) {
                                continue;
                            }
                            if engine.clone().is_protect(k) {
                                render += &format!(
                                    "const {k}: {} = {};\n",
                                    v.type_of().format(),
                                    v.format()
                                );
                            } else {
                                render += &format!("let {k} = {};\n", v.format());
                            }
                        }
                        if let Ok(mut file) = File::create(arg.get_text()?) {
                            if file.write_all(render.as_bytes()).is_ok() {
                                Ok(Type::Text("Environment is saved!".to_string()))
                            } else {
                                Err(Fault::IO)
                            }
                        } else {
                            Err(Fault::IO)
                        }
                    })),
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

    fn parse(source: &str) -> Result<Program, Fault> {
        let mut program: Program = Vec::new();
        for line in tokenize(source, &[';'])? {
            let line = line.trim();
            // Ignore empty line and comment
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            program.push(Statement::parse(line)?);
        }
        Ok(program)
    }

    fn eval(&mut self, program: &Program) -> Result<Type, Fault> {
        let mut result = Type::Null;
        for code in program {
            result = code.eval(self)?
        }
        Ok(result)
    }

    fn alloc(&mut self, name: &String, value: &Type) -> Result<(), Fault> {
        if self.is_protect(name) {
            return Err(Fault::AccessDenied);
        }
        if is_identifier(name) {
            self.env.insert(name.clone(), value.clone());
            Ok(())
        } else {
            Err(Fault::Syntax)
        }
    }

    fn free(&mut self, name: &String) -> Result<(), Fault> {
        if self.is_protect(name) {
            return Err(Fault::AccessDenied);
        }
        ok!(self.env.shift_remove(name), Fault::Refer(name.to_string()))?;
        Ok(())
    }

    fn add_protect(&mut self, name: &String) {
        if !self.is_protect(name) {
            self.protect.push(name.clone());
        }
    }

    fn is_protect(&mut self, name: &String) -> bool {
        self.protect.contains(name)
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
                        engine.alloc(name, &val)?;
                        if *protect {
                            engine.add_protect(name);
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
                        let updated_obj = obj.modify_inside(key, Some(val.clone()))?;
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

    fn parse(code: &str) -> Result<Statement, Fault> {
        let code = code.trim();
        if let Some(code) = code.strip_prefix("print") {
            let mut exprs = vec![];
            for i in tokenize(code, &[','])? {
                exprs.push(Expr::parse(&i)?)
            }
            Ok(Statement::Print(exprs))
        } else if let Some(code) = code.strip_prefix("let") {
            let splited = tokenize(code, &['='])?;
            let (name, code) = (ok!(splited.get(0))?, ok!(splited.get(1))?);
            let splited = tokenize(name, &[':'])?;
            if let (Some(name), Some(sig)) = (splited.get(0), splited.get(1)) {
                Ok(Statement::Let(
                    Expr::parse(name)?,
                    false,
                    Some(Signature::parse(sig)?),
                    Expr::parse(code)?,
                ))
            } else {
                Ok(Statement::Let(
                    Expr::parse(name)?,
                    false,
                    None,
                    Expr::parse(code)?,
                ))
            }
        } else if let Some(code) = code.strip_prefix("const") {
            let (name, code) = ok!(code.split_once("="))?;
            let (name, sig) = ok!(name.split_once(":"))?;
            Ok(Statement::Let(
                Expr::parse(name)?,
                true,
                Some(Signature::parse(sig)?),
                Expr::parse(code)?,
            ))
        } else if let Some(code) = code.strip_prefix("if") {
            let code = tokenize(code, SPACE.as_ref())?;
            if let Some(pos) = code.iter().position(|i| i == "else") {
                Ok(Statement::If(
                    Expr::parse(ok!(code.get(0))?)?,
                    Expr::parse(&ok!(code.get(1..pos))?.join(&SPACE[0].to_string()))?,
                    Some(Expr::Block(Engine::parse(
                        &ok!(code.get(pos + 1..))?.join(&SPACE[0].to_string()),
                    )?)),
                ))
            } else {
                Ok(Statement::If(
                    Expr::parse(ok!(code.get(0))?)?,
                    Expr::parse(&ok!(code.get(1..))?.join(&SPACE[0].to_string()))?,
                    None,
                ))
            }
        } else if let Some(code) = code.strip_prefix("match") {
            let tokens = tokenize(code, SPACE.as_ref())?;
            let expr = Expr::parse(ok!(tokens.get(0))?)?;
            let tokens = tokenize(trim!(ok!(tokens.get(1))?, "{", "}"), &[','])?;
            let mut body = vec![];
            for i in tokens {
                let tokens = tokenize(&i, SPACE.as_ref())?;
                let pos = ok!(tokens.iter().position(|i| i == "=>"))?;
                let mut cond = vec![];
                for i in tokenize(&ok!(tokens.get(..pos))?.join(&SPACE[0].to_string()), &['|'])? {
                    cond.push(Expr::parse(&i)?)
                }
                body.push((
                    cond,
                    Expr::parse(&ok!(tokens.get(pos + 1..))?.join(&SPACE[0].to_string()))?,
                ))
            }
            Ok(Statement::Match(expr, body))
        } else if let Some(code) = code.strip_prefix("for") {
            let code = tokenize(&code, SPACE.as_ref())?;
            if code.get(1).and_then(|x| Some(x == "in")).unwrap_or(false) {
                Ok(Statement::For(
                    Expr::parse(ok!(code.get(0))?)?,
                    Expr::parse(ok!(code.get(2))?)?,
                    Expr::parse(ok!(code.get(3))?)?,
                ))
            } else {
                Err(Fault::Syntax)
            }
        } else if let Some(code) = code.strip_prefix("while") {
            let code = tokenize(code, SPACE.as_ref())?;
            Ok(Statement::While(
                Expr::parse(ok!(code.get(0))?)?,
                Expr::parse(ok!(code.get(1))?)?,
            ))
        } else if let Some(code) = code.strip_prefix("fault") {
            Ok(Statement::Fault(some!(Expr::parse(code))))
        } else {
            Ok(Statement::Return(Expr::parse(code)?))
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
            Statement::Return(expr) => expr.format(),
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
            Expr::Block(block) => engine.eval(block)?,
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
                if name == "_" {
                    Type::Symbol(name.to_string())
                } else if let Some(refer) = engine.env.get(name.as_str()) {
                    refer.clone()
                } else {
                    return Err(Fault::Refer(name.to_string()));
                }
            }
            Expr::Value(value) => value.clone(),
        })
    }

    fn parse(source: &str) -> Result<Expr, Fault> {
        let token_list: Vec<String> = tokenize(source, SPACE.as_ref())?;
        if token_list.len() >= 2 {
            Ok(Expr::Infix(Box::new(Operator::parse(source)?)))
        } else {
            let token = ok!(token_list.last())?.trim().to_string();
            Ok(if let Ok(n) = token.parse::<f64>() {
                Expr::Value(Type::Number(n))
            } else if let Ok(sig) = Signature::parse(&token) {
                Expr::Value(Type::Signature(sig))
            // Prefix operators
            } else if token.starts_with("&") {
                let token = remove!(token, "&");
                Expr::Value(Type::Refer(token))
            } else if token.starts_with("*") {
                let token = remove!(token, "*");
                Expr::Infix(Box::new(Operator::Derefer(Expr::parse(&token)?)))
            } else if token.starts_with("!") {
                let token = remove!(token, "!");
                Expr::Infix(Box::new(Operator::Not(Expr::parse(&token)?)))
            } else if token.starts_with("-") {
                let token = remove!(token, "-");
                Expr::Infix(Box::new(Operator::Sub(
                    Expr::Value(Type::Number(0.0)),
                    Expr::parse(&token)?,
                )))
            } else if token.starts_with("(") && token.ends_with(")") {
                let token = trim!(token, "(", ")");
                Expr::parse(token)?
            } else if token.starts_with("{") && token.ends_with("}") {
                let token = trim!(token, "{", "}");
                Expr::Block(Engine::parse(token)?)
            } else if token.starts_with("@{") && token.ends_with("}") {
                let token = trim!(token, "@{", "}");
                let mut result = Vec::new();
                for i in tokenize(token, &[','])? {
                    let splited = tokenize(&i, &[':'])?;
                    result.push((
                        Expr::parse(ok!(splited.get(0))?)?,
                        Expr::parse(ok!(splited.get(1))?)?,
                    ));
                }
                Expr::Struct(result)
            } else if token.starts_with("[") && token.ends_with("]") {
                let token = trim!(token, "[", "]");
                let mut list = vec![];
                for elm in tokenize(token, &[','])? {
                    list.push(Expr::parse(&elm)?);
                }
                Expr::List(list)
            } else if token.starts_with("\"") && token.ends_with("\"") {
                let text = trim!(token, "\"", "\"");
                Expr::Value(Type::Text(text_escape(text)))
            // Text formating
            } else if token.starts_with("f\"") && token.ends_with('"') {
                let text = trim!(token, "f\"", "\"");
                let splited = text_format(text)?;
                let mut result = Expr::Value(Type::Text(String::new()));
                for i in splited {
                    if i.starts_with("{") && i.ends_with("}") {
                        let i = trim!(i, "{", "}");
                        result = Expr::Infix(Box::new(Operator::Add(
                            result,
                            Expr::Infix(Box::new(Operator::As(
                                Expr::Block(Engine::parse(i)?),
                                Expr::Value(Type::Signature(Signature::Text)),
                            ))),
                        )));
                    } else {
                        result = Expr::Infix(Box::new(Operator::Add(
                            result,
                            Expr::Value(Type::Text(text_escape(&i))),
                        )));
                    }
                }
                result
            // Functionize operator
            } else if token.starts_with("`") && token.ends_with("`") {
                let token = trim!(token, "`", "`");
                let source = format!("λx.λy.(x {token} y)");
                let expr = Expr::parse(&source)?;
                if expr.format() != source {
                    return Err(Fault::Syntax);
                }
                expr
            // Lambda abstract that original formula in the theory
            } else if token.starts_with("λ") && token.contains(".") {
                let token = remove!(token, "λ");
                let (arg, body) = ok!(token.split_once("."))?;
                Expr::Value(Type::Function(Function::UserDefined(
                    arg.to_string(),
                    Box::new(Expr::parse(body)?),
                )))
            // Lambda abstract using back-slash instead of lambda mark
            } else if token.starts_with("\\") && token.contains(".") {
                let token = remove!(token, "\\");
                let (arg, body) = ok!(token.split_once("."))?;
                Expr::Value(Type::Function(Function::UserDefined(
                    arg.to_string(),
                    Box::new(Expr::parse(body)?),
                )))
            // Imperative style syntactic sugar of lambda abstract
            } else if token.starts_with("fn(") && token.contains("->") && token.ends_with(")") {
                let token = trim!(token, "fn(", ")");
                let (args, body) = ok!(token.split_once("->"))?;
                let mut args: Vec<&str> = args.split(",").collect();
                args.reverse();
                let mut func = Expr::Value(Type::Function(Function::UserDefined(
                    ok!(args.first())?.trim().to_string(),
                    Box::new(Expr::Block(Engine::parse(body)?)),
                )));
                // Currying
                for arg in ok!(args.get(1..))? {
                    func = Expr::Value(Type::Function(Function::UserDefined(
                        arg.trim().to_string(),
                        Box::new(func),
                    )));
                }
                func
            // Imperative style syntactic sugar of list access by index
            } else if token.contains('[') && token.ends_with(']') {
                let token = trim!(token, "", "]");
                let (name, args) = ok!(token.split_once("["))?;
                Expr::Infix(Box::new(Operator::Access(
                    Expr::parse(name.trim())?,
                    Expr::parse(args)?,
                )))
            // Imperative style syntactic sugar of function application
            } else if token.contains('(') && token.ends_with(')') {
                let token = trim!(token, "", ")");
                let (name, args) = ok!(token.split_once("("))?;
                let is_lazy = name.ends_with("?");
                let args = tokenize(args, &[','])?;
                let mut call = Expr::Infix(Box::new(Operator::Apply(
                    Expr::parse(if is_lazy { trim!(name, "", "?") } else { name })?,
                    is_lazy,
                    Expr::parse(ok!(args.first())?)?,
                )));
                for arg in ok!(args.get(1..))? {
                    call = Expr::Infix(Box::new(Operator::Apply(call, is_lazy, Expr::parse(arg)?)));
                }
                call
            // Object-oriented style syntactic sugar of access operator
            } else if token.matches(".").count() >= 1 {
                let (obj, key) = ok!(token.rsplit_once("."))?;
                Expr::Infix(Box::new(Operator::Access(
                    Expr::parse(obj)?,
                    Expr::Value(Type::Text(key.trim().to_string())),
                )))
            } else if token == "null" {
                Expr::Value(Type::Null)
            } else if is_identifier(&token) {
                Expr::Value(Type::Symbol(token))
            } else {
                return Err(Fault::Syntax);
            })
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
                            *protect,
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
                    Type::Text(lhs.clone() + rhs)
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
                } else {
                    lhs.modify_inside(rhs, None)?
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
                    let text = char_vec!(text);
                    Type::Text(ok!(
                        text.get(index as usize).cloned(),
                        Fault::Index(rhs.clone(), lhs.clone())
                    )?)
                } else if let (Type::Struct(st), Type::Text(index)) = (lhs.clone(), rhs.clone()) {
                    ok!(st.get(&index), Fault::Key(rhs, lhs))?.clone()
                } else if let (Type::List(list), Type::Range(start, end)) =
                    (lhs.clone(), rhs.clone())
                {
                    let mut result = vec![];
                    for i in start..end {
                        result.push(ok!(
                            list.get(i).cloned(),
                            Fault::Index(rhs.clone(), lhs.clone())
                        )?);
                    }
                    Type::List(result)
                } else if let (Type::Text(text), Type::Range(start, end)) =
                    (lhs.clone(), rhs.clone())
                {
                    let mut result = String::new();
                    let text: Vec<char> = text.chars().collect();
                    for i in start..end {
                        result.push(ok!(
                            text.get(i).cloned(),
                            Fault::Index(rhs.clone(), lhs.clone())
                        )?);
                    }
                    Type::Text(result)
                } else if let (Type::List(list), Type::List(query)) = (lhs.clone(), rhs.clone()) {
                    let index = ok!(
                        list.windows(query.len())
                            .position(|i| Type::List(i.to_vec()).format()
                                == Type::List(query.clone()).format()),
                        Fault::Key(rhs, lhs)
                    )?;
                    Type::Range(index, index + query.len())
                } else if let (Type::Text(text), Type::Text(query)) = (lhs.clone(), rhs.clone()) {
                    let index = ok!(text.find(&query), Fault::Key(rhs, lhs))?;
                    Type::Range(index, index + query.chars().count())
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
                lhs.cast(&rhs.get_signature()?, engine)?
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
            Operator::AssignAdd(name, rhs) => Operator::Assign(
                name.to_owned(),
                Expr::Infix(Box::new(Operator::Add(name.to_owned(), rhs.clone()))),
            )
            .eval(engine)?,
            Operator::AssignSub(name, rhs) => Operator::Assign(
                name.to_owned(),
                Expr::Infix(Box::new(Operator::Sub(name.to_owned(), rhs.clone()))),
            )
            .eval(engine)?,
            Operator::AssignMul(name, rhs) => Operator::Assign(
                name.to_owned(),
                Expr::Infix(Box::new(Operator::Mul(name.to_owned(), rhs.clone()))),
            )
            .eval(engine)?,
            Operator::AssignDiv(name, rhs) => Operator::Assign(
                name.to_owned(),
                Expr::Infix(Box::new(Operator::Div(name.to_owned(), rhs.clone()))),
            )
            .eval(engine)?,
            Operator::AssignMod(name, rhs) => Operator::Assign(
                name.to_owned(),
                Expr::Infix(Box::new(Operator::Mod(name.to_owned(), rhs.clone()))),
            )
            .eval(engine)?,
            Operator::AssignPow(name, rhs) => Operator::Assign(
                name.to_owned(),
                Expr::Infix(Box::new(Operator::Pow(name.to_owned(), rhs.clone()))),
            )
            .eval(engine)?,
            Operator::To(from, to) => {
                let from = from.eval(engine)?.get_number()? as usize;
                let to = to.eval(engine)?.get_number()? as usize;
                if from < to {
                    Type::Range(from, to)
                } else {
                    return Err(Fault::Syntax);
                }
            }
        })
    }

    fn parse(source: &str) -> Result<Operator, Fault> {
        let token_list: Vec<String> = tokenize(source, SPACE.as_ref())?;
        let token = Expr::parse(ok!(token_list.last())?)?;
        let operator = ok!(token_list.get(ok!(token_list.len().checked_sub(2))?))?;
        let has_lhs = |len: usize| {
            Expr::parse(&ok!(token_list.get(..token_list.len() - len))?.join(&SPACE[0].to_string()))
        };
        Ok(match operator.as_str() {
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
                            Expr::parse(&operator)?,
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

    #[error("index `{}` is out of the sequence `{}`", _0.format(), _1.format())]
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
    Range(usize, usize),
    Function(Function),
    Signature(Signature),
    Struct(IndexMap<String, Type>),
    Null,
}

impl Type {
    fn cast(&self, sig: &Signature, engine: &mut Engine) -> Result<Type, Fault> {
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
                Type::Struct(st) => {
                    if let Some(Type::Function(as_number)) = st.get("asNumber") {
                        Operator::Apply(
                            Expr::Value(Type::Function(as_number.clone())),
                            false,
                            Expr::Value(self.clone()),
                        )
                        .eval(engine)?
                        .get_number()?
                    } else {
                        return err;
                    }
                }
                _ => return err,
            }),
            Signature::Text => Type::Text(match self {
                Type::Symbol(s) | Type::Text(s) => s.to_string(),
                Type::Null => String::new(),
                Type::Struct(st) => {
                    if let Some(Type::Function(as_text)) = st.get("asText") {
                        Operator::Apply(
                            Expr::Value(Type::Function(as_text.clone())),
                            false,
                            Expr::Value(self.clone()),
                        )
                        .eval(engine)?
                        .get_text()?
                    } else {
                        self.format()
                    }
                }
                _ => self.format(),
            }),
            Signature::List => Type::List(match self {
                Type::List(list) => list.to_owned(),
                Type::Text(text) => text.chars().map(|i| Type::Text(i.to_string())).collect(),
                Type::Struct(strct) => strct
                    .iter()
                    .map(|(k, y)| Type::List(vec![Type::Text(k.to_owned()), y.to_owned()]))
                    .collect(),
                Type::Range(start, end) => {
                    let mut range: Vec<Type> = vec![];
                    let mut current = *start;
                    while current < *end {
                        range.push(Type::Number(current as f64));
                        current += 1;
                    }
                    range
                }
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
            Type::Range(start, end) => format!("({start} ~ {end})",),
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
            Type::Range(_, _) => Signature::Range,
            Type::Signature(_) => Signature::Signature,
            Type::Function(_) => Signature::Function,
            Type::Struct(st) => {
                if let Some(r#type) = st.get("class") {
                    r#type.get_signature().unwrap_or(Signature::Struct)
                } else {
                    Signature::Struct
                }
            }
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
        } else if pattern.format() == "_" {
            true
        } else {
            self.format() == pattern.format()
        }
    }

    fn modify_inside(&self, index: Type, val: Option<Type>) -> Result<Type, Fault> {
        Ok(
            if let (Type::List(mut list), Type::Number(index)) = (self.clone(), index.clone()) {
                index_check!(list, index, self);
                if let Some(val) = val {
                    list[index as usize] = val.clone();
                } else {
                    list.remove(index as usize);
                }
                Type::List(list)
            } else if let (Type::Text(text), Type::Number(index)) = (self.clone(), index.clone()) {
                let mut text = char_vec!(text);
                index_check!(text, index, self);
                if let Some(val) = val {
                    text[index as usize] = val.get_text()?;
                } else {
                    text.remove(index as usize);
                }
                Type::Text(text.concat())
            } else if let (Type::Struct(mut st), Type::Text(index)) = (self.clone(), index.clone())
            {
                if let Some(val) = val {
                    st.insert(index, val.clone());
                } else {
                    st.shift_remove(&index);
                }
                Type::Struct(st)
            } else if let (Type::List(mut list), Type::Range(start, end)) =
                (self.clone(), index.clone())
            {
                for _ in start..end {
                    index_check!(list, start as f64, self);
                    list.remove(start);
                }
                if let Some(val) = val {
                    list.insert(start, val.clone());
                }
                Type::List(list)
            } else if let (Type::Text(text), Type::Range(start, end)) =
                (self.clone(), index.clone())
            {
                let mut text = char_vec!(text);
                for _ in start..end {
                    index_check!(text, start as f64, self);
                    text.remove(start);
                }
                if let Some(val) = val {
                    text.insert(start, val.get_text()?);
                }
                Type::Text(text.concat())
            } else if let (Type::Text(text), Type::Text(index)) = (&self, &index) {
                if let Some(val) = val {
                    Type::Text(text.replace(index, &val.get_text()?))
                } else {
                    Type::Text(remove!(text, index))
                }
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
    Range,
    Function,
    Signature,
    Struct,
    Class(String),
}

impl Signature {
    fn parse(token: &str) -> Result<Signature, Fault> {
        let token = token.trim();
        Ok(if token == "number" {
            Signature::Number
        } else if token == "symbol" {
            Signature::Symbol
        } else if token == "text" {
            Signature::Text
        } else if token == "refer" {
            Signature::Refer
        } else if token == "list" {
            Signature::List
        } else if token == "range" {
            Signature::Range
        } else if token == "function" {
            Signature::Function
        } else if token == "signature" {
            Signature::Signature
        } else if token == "struct" {
            Signature::Struct
        } else if token.starts_with("#") {
            let token = trim!(token, "#", "");
            if is_identifier(token) {
                Signature::Class(token.to_string())
            } else {
                return Err(Fault::Syntax);
            }
        } else {
            return Err(Fault::Syntax);
        })
    }

    fn format(&self) -> String {
        match self {
            Signature::Number => "number".to_string(),
            Signature::Symbol => "symbol".to_string(),
            Signature::Text => "text".to_string(),
            Signature::Refer => "refer".to_string(),
            Signature::List => "list".to_string(),
            Signature::Range => "range".to_string(),
            Signature::Function => "function".to_string(),
            Signature::Signature => "signature".to_string(),
            Signature::Struct => "struct".to_string(),
            Signature::Class(s) => format!("#{s}"),
        }
    }
}

fn tokenize(input: &str, delimiter: &[char]) -> Result<Vec<String>, Fault> {
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
                    if in_parentheses != 0 {
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

fn is_identifier(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    if name == "_" {
        return true;
    }
    let mut chars = name.chars();
    let first_char = chars.next().unwrap();
    if !UnicodeXID::is_xid_start(first_char) {
        return false;
    }
    if !chars.all(UnicodeXID::is_xid_continue) {
        return false;
    }
    if RESERVED.contains(&name) {
        return false;
    }
    true
}

fn text_format(input: &str) -> Result<Vec<String>, Fault> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current_token = String::new();
    let mut in_parentheses: usize = 0;
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
                '{' => {
                    if in_parentheses == 0 {
                        if !current_token.is_empty() {
                            tokens.push(current_token.clone());
                        }
                        current_token = c.to_string();
                    } else {
                        current_token.push(c)
                    }
                    in_parentheses += 1;
                }
                '}' => {
                    current_token.push(c);
                    if in_parentheses != 0 {
                        in_parentheses -= 1;
                    } else {
                        return Err(Fault::Syntax);
                    }
                    if in_parentheses == 0 {
                        if !current_token.is_empty() {
                            tokens.push(current_token.clone());
                        }
                        current_token.clear();
                    }
                }
                '\\' => {
                    current_token.push(c);
                    is_escape = true;
                }
                _ => current_token.push(c),
            }
        }
    }

    // Syntax error check
    if is_escape || in_parentheses != 0 {
        return Err(Fault::Syntax);
    }
    if !current_token.is_empty() {
        tokens.push(current_token.clone());
        current_token.clear();
    }
    Ok(tokens)
}

fn text_escape(text: &str) -> String {
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
