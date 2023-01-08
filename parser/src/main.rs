#[derive(Debug, Clone, PartialEq, Eq, Hash)]

/**
 * 位置情報
 * NOTE: Loc(4,6)なら入力文字の5文字目から7文字目までの区間を表す(0始まり)
 */
struct Loc(usize, usize);
impl Loc {
    fn merge(&self, other: &Loc) -> Loc {
        use std::cmp::{max, min};
        Loc(min(self.0, other.0), max(self.1, other.1))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
// 値と位置情報を持つ構造体
struct Annot<T> {
    value: T,
    loc: Loc,
}
impl<T> Annot<T> {
    fn new(value: T, loc: Loc) -> Self {
        Self { value, loc }
    }
}

/**
 * トークンの実装
 * トークンはトークンの種類に位置情報を加えたもの
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // [0-9][0-9]*
    Number(u64),
    // +
    Plus,
    // -
    Minus,
    // *
    Asterisk,
    // /
    Slash,
    // (
    LParen,
    // )
    RParen,
}

type Token = Annot<TokenKind>;

// 関連関数を定義
impl Token {
    // 数値(0~9)の場合は数値と位置情報を持つ。それ以外は位置情報のみ持つ
    fn number(n: u64, loc: Loc) -> Self {
        Self::new(TokenKind::Number(n), loc)
    }
    fn plus(loc: Loc) -> Self {
        Self::new(TokenKind::Plus, loc)
    }
    fn minus(loc: Loc) -> Self {
        Self::new(TokenKind::Minus, loc)
    }
    fn asterisk(loc: Loc) -> Self {
        Self::new(TokenKind::Asterisk, loc)
    }
    fn slash(loc: Loc) -> Self {
        Self::new(TokenKind::Slash, loc)
    }
    fn lparen(loc: Loc) -> Self {
        Self::new(TokenKind::LParen, loc)
    }
    fn rparen(loc: Loc) -> Self {
        Self::new(TokenKind::RParen, loc)
    }
}

/**
 * 字句解析器の実装
 * 戻り値をトークン列に加え、位置情報を更新
 */

/**
 * posのバイドが期待するものであれば1バイトを消費してposを1進める
 * input 入力文字列
 * pos 入力バイト
 * b 期待するバイト
 */
fn consume_byte(input: &[u8], pos: usize, b: u8) -> Result<(u8, usize), LexError> {
    // posが入力サイズ以上なら入力が終わっている。
    // 1バイト期待しているのに終わっているのでエラー
    if input.len() <= pos {
        return Err(LexError::eof(Loc(pos, pos)));
    }
    // 入力が期待するものでなければエラー
    if input[pos] != b {
        return Err(LexError::invalid_char(
            input[pos] as char,
            Loc(pos, pos + 1),
        ));
    }

    Ok((b, pos + 1))
}

// 一文字記号を解析する関数
// `Result::map` を使うことで結果が正常だった場合の処理を簡潔に書く。
// 下記のコードと同義
// ```
// match consume_byte(input, start, b'+') {
//     // Okではconsume_byteの返り値を引数にとる
//     Ok((_, end)) => (Token::plus(Loc(start, end)), end),
//     Err(err) => Err(err),
// }
// ```
fn lex_plus(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'+').map(|(_, end)| (Token::plus(Loc(start, end)), end))
}
fn lex_minus(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'-').map(|(_, end)| (Token::minus(Loc(start, end)), end))
}
fn lex_asterisk(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'*').map(|(_, end)| (Token::asterisk(Loc(start, end)), end))
}
fn lex_slash(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'/').map(|(_, end)| (Token::slash(Loc(start, end)), end))
}
fn lex_lparen(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b'(').map(|(_, end)| (Token::lparen(Loc(start, end)), end))
}
fn lex_rparen(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b')').map(|(_, end)| (Token::rparen(Loc(start, end)), end))
}
fn lex_number(input: &[u8], pos: usize) -> Result<(Token, usize), LexError> {
    use std::str::from_utf8;

    let start = pos;
    // 入力に数字が続く限り位置を進める
    let end = recognize_many(input, start, |b| b"1234567890".contains(&b));

    // 数字の列を数値に変換
    let n = from_utf8(&input[start..end])
        // start..posの構成から `from_utf8` は常に成功するため`unwrap`しても安全
        .unwrap()
        .parse()
        // 同じく構成から `parse` は常に成功する
        .unwrap();
    Ok((Token::number(n, Loc(start, end)), end))
}
fn skip_spaces(input: &[u8], pos: usize) -> Result<((), usize), LexError> {
    // 入力に空白が続く限り位置を進める
    let pos = recognize_many(input, pos, |b| b" \n\t".contains(&b));
    Ok(((), pos))
}

// 条件に当てはまる入力を複数認識して最終的には位置情報を返す
fn recognize_many(input: &[u8], mut pos: usize, mut f: impl FnMut(u8) -> bool) -> usize {
    while pos < input.len() && f(input[pos]) {
        pos += 1;
    }
    pos
}

/**
 * 字句解析器
 */
fn lex(input: &str) -> Result<Vec<Token>, LexError> {
    // 解析結果を保存するベクタ
    let mut tokens = Vec::new();
    // 入力
    let input = input.as_bytes();
    // 位置を管理する値
    let mut pos = 0;
    // サブレキサを呼んだ後`pos`を更新するマクロ
    macro_rules! lex_a_token {
        ($lexer:expr) => {{
            let (tok, p) = $lexer?;
            tokens.push(tok);
            pos = p;
        }};
    }
    while pos < input.len() {
        // ここでそれぞれの関数に`input`と`pos`を渡す
        match input[pos] {
            // 遷移図通りの実装
            b'0'..=b'9' => lex_a_token!(lex_number(input, pos)),
            b'+' => lex_a_token!(lex_plus(input, pos)),
            b'-' => lex_a_token!(lex_minus(input, pos)),
            b'*' => lex_a_token!(lex_asterisk(input, pos)),
            b'/' => lex_a_token!(lex_slash(input, pos)),
            b'(' => lex_a_token!(lex_lparen(input, pos)),
            b')' => lex_a_token!(lex_rparen(input, pos)),
            // 空白を扱う
            b' ' | b'\n' | b'\t' => {
                let ((), p) = skip_spaces(input, pos)?;
                pos = p;
            }
            // それ以外がくるとエラー
            b => return Err(LexError::invalid_char(b as char, Loc(pos, pos + 1))),
        }
    }
    Ok(tokens)
}

// 単項演算子を表すデータ型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum UniOpKind {
    Plus,
    Minus,
}
type UniOp = Annot<UniOpKind>;
impl UniOp {
    fn plus(loc: Loc) -> Self {
        Self::new(UniOpKind::Plus, loc)
    }
    fn minus(loc: Loc) -> Self {
        Self::new(UniOpKind::Minus, loc)
    }
}

// 二項演算子を表すデータ型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum BinOpKind {
    Add,
    Sub,
    Mult,
    Div,
}
type BinOp = Annot<BinOpKind>;
impl BinOp {
    fn add(loc: Loc) -> Self {
        Self::new(BinOpKind::Add, loc)
    }
    fn sub(loc: Loc) -> Self {
        Self::new(BinOpKind::Sub, loc)
    }
    fn mult(loc: Loc) -> Self {
        Self::new(BinOpKind::Mult, loc)
    }
    fn div(loc: Loc) -> Self {
        Self::new(BinOpKind::Div, loc)
    }
}

// ASTを表すデータ型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AstKind {
    Num(u64),
    // 単項演算
    UniOp { op: UniOp, e: Box<Ast> },
    // 二項演算
    BinOp { op: BinOp, l: Box<Ast>, r: Box<Ast> },
}

type Ast = Annot<AstKind>;
impl Ast {
    fn num(n: u64, loc: Loc) -> Self {
        // impl<T> Annot<T>で実装したnewを呼ぶ
        Self::new(AstKind::Num(n), loc)
    }

    fn uniop(op: UniOp, e: Ast, loc: Loc) -> Self {
        Self::new(AstKind::UniOp { op, e: Box::new(e) }, loc)
    }

    fn binop(op: BinOp, l: Ast, r: Ast, loc: Loc) -> Self {
        Self::new(
            AstKind::BinOp {
                op,
                l: Box::new(l),
                r: Box::new(r),
            },
            loc,
        )
    }
}

/**
 * 構文解析器の実装
 * EXPR = EXPR3
 * EXPR3 = EXPR3, ("+" | "-"), EXPR2 | EXPR2;
 * EXPR2 = EXPR2, ("*" | "/"), EXPR1 | EXPR1;
 * EXPR1 = ("+" | "-"), ATOM | ATOM;
 * ATOM = UNUMBER | "(" , EXPR3, ")";
 * UNUMBER = DIGIT, {DIGIT};
 * DIGIT = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7"  | "8" | "9";
 */
use std::iter::Peekable;
fn parse(tokens: Vec<Token>) -> Result<Ast, ParseError> {
    // イテレータにして、次の値を覗き見可能にする(peekable)
    let mut tokens = tokens.into_iter().peekable();

    // parse_exprを読んでエラー処理
    let ret = parse_expr(&mut tokens)?;
    match tokens.next() {
        Some(tok) => Err(ParseError::RedundantExpression(tok)),
        None => Ok(ret),
    }
}

// parse_expr3とparse_expr2内で呼び出すを共通化
fn parse_left_binop<Tokens>(
    tokens: &mut Peekable<Tokens>,
    subexpr_parser: fn(&mut Peekable<Tokens>) -> Result<Ast, ParseError>,
    op_parser: fn(&mut Peekable<Tokens>) -> Result<BinOp, ParseError>,
) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    let mut e = subexpr_parser(tokens)?;
    loop {
        match tokens.peek() {
            Some(_) => {
                let op = match op_parser(tokens) {
                    Ok(op) => op,
                    // ここでパースに失敗したのはこれ以上中置演算子がないという意味
                    Err(_) => break,
                };
                let r = subexpr_parser(tokens)?;
                let loc = e.loc.merge(&r.loc);
                e = Ast::binop(op, e, r, loc)
            }
            _ => break,
        }
    }
    Ok(e)
}

// EXPRの実装
fn parse_expr<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    parse_expr3(tokens)
}

// EXPR3の実装
fn parse_expr3<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    // EXPR2をパース
    let mut e = parse_expr2(tokens)?;
    // EXPR3_Loop
    loop {
        match tokens.peek().map(|tok| tok.value) {
            // `("+" | "-")`である事の確認
            Some(TokenKind::Plus) | Some(TokenKind::Minus) => {
                let op = match tokens.next().unwrap() {
                    Token {
                        value: TokenKind::Plus,
                        loc,
                    } => BinOp::add(loc),
                    Token {
                        value: TokenKind::Minus,
                        loc,
                    } => BinOp::sub(loc),
                    // `("+" | "-")`である事の確認をしたのでそれ以外な存在しない
                    _ => unreachable!(),
                };
                // EXPR2をパース
                let r = parse_expr2(tokens)?;
                let loc = e.loc.merge(&r.loc);
                e = Ast::binop(op, e, r, loc);
            }
            _ => return Ok(e),
        }
    }
}

// EXPR2の実装
fn parse_expr2<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    // EXPR1をパース
    let mut e = parse_expr1(tokens)?;
    // EXPR2
    loop {
        match tokens.peek().map(|tok| tok.value) {
            // `("*" | "/")`である事の確認
            Some(TokenKind::Asterisk) | Some(TokenKind::Slash) => {
                let op = match tokens.next().unwrap() {
                    Token {
                        value: TokenKind::Asterisk,
                        loc,
                    } => BinOp::mult(loc),
                    Token {
                        value: TokenKind::Slash,
                        loc,
                    } => BinOp::div(loc),
                    // `("*" | "/")`である事の確認をしたのでそれ以外な存在しない
                    _ => unreachable!(),
                };
                // EXPR1をパース
                let r = parse_expr1(tokens)?;
                let loc = e.loc.merge(&r.loc);
                e = Ast::binop(op, e, r, loc);
            }
            _ => return Ok(e),
        }
    }
}

// EXPR1の実装
fn parse_expr1<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    match tokens.peek().map(|tok| tok.value) {
        Some(TokenKind::Plus) | Some(TokenKind::Minus) => {
            // `("+" | "-")`
            let op = match tokens.next() {
                Some(Token {
                    value: TokenKind::Plus,
                    loc,
                }) => UniOp::plus(loc),
                Some(Token {
                    value: TokenKind::Minus,
                    loc,
                }) => UniOp::minus(loc),
                _ => unreachable!(),
            };
            // ATOM
            let e = parse_atom(tokens)?;
            let loc = op.loc.merge(&e.loc);
            Ok(Ast::uniop(op, e, loc))
        }
        // `| ATOM`
        _ => parse_atom(tokens),
    }
}

// ATOMの実装
fn parse_atom<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where
    Tokens: Iterator<Item = Token>,
{
    tokens
        .next()
        .ok_or(ParseError::Eof)
        .and_then(|tok| match tok.value {
            // UNUMBER
            TokenKind::Number(n) => Ok(Ast::new(AstKind::Num(n), tok.loc)),

            // "(", EXPR3, ")"
            TokenKind::LParen => {
                let e = parse_expr(tokens)?;
                match tokens.next() {
                    Some(Token {
                        value: TokenKind::RParen,
                        ..
                    }) => Ok(e),
                    Some(t) => Err(ParseError::RedundantExpression(t)),
                    _ => Err(ParseError::UnclosedOpenParen(tok)),
                }
            }
            _ => Err(ParseError::NotExpression(tok)),
        })
}

use std::str::FromStr;
// AstにFromStrを実装すると`str::parse`が使える
impl FromStr for Ast {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // 字句解析 -> 構文解析の順に実行する
        let tokens = lex(s)?;
        let ast = parse(tokens)?;
        Ok(ast)
    }
}

/**
 * エラーの定義
 */
// lexer
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LexErrorKind {
    InvalidChar(char),
    Eof,
}

type LexError = Annot<LexErrorKind>;
impl LexError {
    fn invalid_char(c: char, loc: Loc) -> Self {
        LexError::new(LexErrorKind::InvalidChar(c), loc)
    }
    fn eof(loc: Loc) -> Self {
        LexError::new(LexErrorKind::Eof, loc)
    }
}

// perser
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ParseError {
    // 予期しないトークンがきた場合
    UnexpectedToken(Token),

    /** 期待していないものが来た場合 */
    // 式を期待していた場合
    NotExpression(Token),
    // 演算子を期待していた場合
    NotOperator(Token),

    // 括弧が閉じられていない場合
    UnclosedOpenParen(Token),

    // 式の解析が終わったのにトークンが残っている
    RedundantExpression(Token),

    // パース途中で入力が終わった
    Eof,
}

// interpreter
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum InterpreterErrorKind {
    DivisionByZero,
}
type InterpreterError = Annot<InterpreterErrorKind>;
impl InterpreterError {
    fn show_diagnostic(&self, input: &str) {
        // エラー情報を簡単に表示し
        eprintln!("{}", self);
        // エラー位置を指示する
        print_annot(input, self.loc.clone());
    }
}

// 字句解析エラーと構文解析エラーを統合するエラー型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Error {
    Lexer(LexError),
    Parser(ParseError),
}

impl From<LexError> for Error {
    fn from(e: LexError) -> Self {
        Error::Lexer(e)
    }
}

impl From<ParseError> for Error {
    fn from(e: ParseError) -> Self {
        Error::Parser(e)
    }
}

/**
 * エラーハンドリング
 */

// 字句解析
impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::LexErrorKind::*;
        let loc = &self.loc;
        match self.value {
            InvalidChar(c) => write!(f, "{}: invalid char '{}'", loc, c),
            Eof => write!(f, "End of file"),
        }
    }
}

// 構文解析
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ParseError::*;
        match self {
            UnexpectedToken(tok) => write!(f, "{}: {} is not expected", tok.loc, tok.value),
            NotExpression(tok) => write!(
                f,
                "{}: '{}' is not a start of expression",
                tok.loc, tok.value
            ),
            NotOperator(tok) => write!(f, "{}: '{}' is not an operator", tok.loc, tok.value),
            UnclosedOpenParen(tok) => write!(f, "{}: '{}' is not closed", tok.loc, tok.value),
            RedundantExpression(tok) => write!(
                f,
                "{}: expression after '{}' is redundant",
                tok.loc, tok.value
            ),
            Eof => write!(f, "End of file"),
        }
    }
}

// インタプリタ
impl fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::InterpreterErrorKind::*;
        match self.value {
            DivisionByZero => write!(f, "division by zero"),
        }
    }
}

/** DisplayとErrorをエラー関連の方に実装 */
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "parser error")
    }
}

impl StdError for InterpreterError {
    fn description(&self) -> &str {
        use self::InterpreterErrorKind::*;
        match self.value {
            DivisionByZero => "the right hand expression of the division evaluates to zero",
        }
    }
}

// 標準ライブラリのErrorを実装(descriptionとcauseは使わず、sourceだけを実装)
// Errorデータ型と名前が重複するのでStdErrorとして導入
use std::error::Error as StdError;
impl StdError for LexError {}
impl StdError for ParseError {}
impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        use self::Error::*;
        match self {
            Lexer(lex) => Some(lex),
            Parser(parse) => Some(parse),
        }
    }
}

/**
 * エラーの実装
 */
impl Error {
    /// 診断メッセージを表示する
    fn show_diagnostic(&self, input: &str) {
        use self::Error::*;
        use self::ParseError as P;
        // エラー情報とその位置情報を取り出す
        let (e, loc): (&dyn StdError, Loc) = match self {
            Lexer(e) => (e, e.loc.clone()),
            Parser(e) => {
                let loc = match e {
                    P::UnexpectedToken(Token { loc, .. })
                    | P::NotExpression((Token { loc, .. }))
                    | P::NotOperator((Token { loc, .. }))
                    | P::UnclosedOpenParen((Token { loc, .. })) => loc.clone(),
                    // redundant expressionはトークン移行行末までがあまりなのでlocの終了位置を調節
                    P::RedundantExpression(Token { loc, .. }) => Loc(loc.0, input.len()),
                    // Eofは位置情報を持っていないのでその場で作る
                    P::Eof => Loc(input.len(), input.len() + 1),
                };
                (e, loc)
            }
        };
        // エラー情報を表示
        eprintln!("{}", e);
        // エラー位置情報を強調して表示
        print_annot(input, loc);
    }
}

/**
 *
 */

/** 文字の表示 */
use std::fmt;
impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::TokenKind::*;
        match self {
            Number(n) => n.fmt(f),
            Plus => write!(f, "+"),
            Minus => write!(f, "-"),
            Asterisk => write!(f, "*"),
            Slash => write!(f, "/"),
            LParen => write!(f, "("),
            RParen => write!(f, ")"),
        }
    }
}

/** 位置情報の表示 */
impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.0, self.1)
    }
}
fn print_annot(input: &str, loc: Loc) {
    // 入力に対して位置情報をわかりやすく示す
    eprintln!("{}", input);
    eprintln!("{}{}", " ".repeat(loc.0), "^".repeat(loc.1 - loc.0));
}

/** エラー詳細表示 */
fn show_trace<E: StdError>(e: E) {
    // エラーがあった場合、そのエラーとsourceを全て出力
    eprintln!("{}", e);
    let mut source = e.source();

    // sourceを全て辿って表示
    while let Some(e) = source {
        eprintln!("caused by {}", e);
        source = e.source()
    }
}

/**
 * インタプリタの作成
 */
// データ型
struct Interpreter;
impl Interpreter {
    pub fn new() -> Self {
        Interpreter
    }

    pub fn eval(&mut self, expr: &Ast) -> Result<i64, InterpreterError> {
        use self::AstKind::*;
        match expr.value {
            Num(n) => Ok(n as i64),
            UniOp { ref op, ref e } => {
                let e = self.eval(e)?;
                Ok(self.eval_uniop(op, e))
            }
            BinOp {
                ref op,
                ref l,
                ref r,
            } => {
                let l = self.eval(l)?;
                let r = self.eval(r)?;
                self.eval_binop(op, l, r)
                    .map_err(|e| InterpreterError::new(e, expr.loc.clone()))
            }
        }
    }
    // 単項演算
    fn eval_uniop(&mut self, op: &UniOp, n: i64) -> i64 {
        use self::UniOpKind::*;
        match op.value {
            Plus => n,
            Minus => -n,
        }
    }
    // 二項演算
    fn eval_binop(&mut self, op: &BinOp, l: i64, r: i64) -> Result<i64, InterpreterErrorKind> {
        use self::BinOpKind::*;
        match op.value {
            Add => Ok(l + r),
            Sub => Ok(l - r),
            Mult => Ok(l * r),
            Div => {
                if r == 0 {
                    Err(InterpreterErrorKind::DivisionByZero)
                } else {
                    Ok(l / r)
                }
            }
        }
    }
}

/**
 * 入力の受付
 */
use std::io;
// プロンプトを表示しユーザーの入力を促す
fn prompt(s: &str) -> io::Result<()> {
    use std::io::{stdout, Write};

    let stdout = stdout();
    let mut stdout = stdout.lock();
    stdout.write(s.as_bytes());
    stdout.flush()
}

fn main() {
    use std::io::{stdin, BufRead, BufReader};

    // インタプリタを用意
    let mut interpreter = Interpreter::new();

    let stdin = stdin();
    let stdin = stdin.lock();
    let stdin = BufReader::new(stdin);
    let mut lines = stdin.lines();

    loop {
        prompt("> ").unwrap();
        // 入力値の取得
        if let Some(Ok(line)) = lines.next() {
            let ast = match line.parse::<Ast>() {
                Ok(ast) => ast,
                Err(e) => {
                    e.show_diagnostic(&line);
                    show_trace(e);
                    continue;
                }
            };
            // インタプリタでevalする
            let n = match interpreter.eval(&ast) {
                Ok(n) => n,
                Err(e) => {
                    e.show_diagnostic(&line);
                    show_trace(e);
                    continue;
                }
            };
            println!("{}", n);
        } else {
            break;
        }
    }
}

/**
 * tests
 */
#[test]
fn test_lexer() {
    assert_eq!(
        lex("1 + 2 * 3 - -10"),
        Ok(vec![
            Token::number(1, Loc(0, 1)),
            Token::plus(Loc(2, 3)),
            Token::number(2, Loc(4, 5)),
            Token::asterisk(Loc(6, 7)),
            Token::number(3, Loc(8, 9)),
            Token::minus(Loc(10, 11)),
            Token::minus(Loc(12, 13)),
            Token::number(10, Loc(13, 15)),
        ])
    )
}

#[test]
fn test_parser() {
    // 1 + 2 * 3 - -10
    let ast = parse(vec![
        Token::number(1, Loc(0, 1)),
        Token::plus(Loc(2, 3)),
        Token::number(2, Loc(4, 5)),
        Token::asterisk(Loc(6, 7)),
        Token::number(3, Loc(8, 9)),
        Token::minus(Loc(10, 11)),
        Token::minus(Loc(12, 13)),
        Token::number(10, Loc(13, 15)),
    ]);
    assert_eq!(
        ast,
        Ok(Ast::binop(
            BinOp::sub(Loc(10, 11)),
            Ast::binop(
                BinOp::add(Loc(2, 3)),
                Ast::num(1, Loc(0, 1)),
                Ast::binop(
                    BinOp::new(BinOpKind::Mult, Loc(6, 7)),
                    Ast::num(2, Loc(4, 5)),
                    Ast::num(3, Loc(8, 9)),
                    Loc(4, 9)
                ),
                Loc(0, 9),
            ),
            Ast::uniop(
                UniOp::minus(Loc(12, 13)),
                Ast::num(10, Loc(13, 15)),
                Loc(12, 15)
            ),
            Loc(0, 15)
        ))
    )
}
