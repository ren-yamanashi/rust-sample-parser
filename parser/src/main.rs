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
enum TokenKind {
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
 * エラーの定義
 */
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum LexErrorKind {
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

    let stdin = stdin();
    let stdin = stdin.lock();
    let stdin = BufReader::new(stdin);
    let mut lines = stdin.lines();

    loop {
        prompt("> ").unwrap();
        // 入力値の取得
        if let Some(Ok(line)) = lines.next() {
            // 字句解析
            let token = lex(&line);
            println!("{:?}", token);
        } else {
            break;
        }
    }
}
