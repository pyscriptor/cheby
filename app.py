from flask import Flask, render_template_string, request
from sympy import kronecker_symbol, nextprime
from sympy.ntheory.primetest import is_square
import ast
import operator
import waitress

# --- Start of code from cheby.py ---

# map AST operators to Python functions
_binops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.floordiv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow
}
_unops = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg
}

def parse_int_expr(expr: str) -> int:
    """
    Safely parse and evaluate an integer expression supporting:
      - integers
      - +, -, *, //, %, ** - parentheses
      - caret (^) as exponent shorthand
    """
    # replace caret with Python exponent operator
    expr = expr.replace('^', '**')
    node = ast.parse(expr, mode='eval')

    def _evaluate(node):
        if isinstance(node, ast.Expression):
            return _evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return node.value
            raise ValueError(f"Non-integer literal {node.value}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _binops:
                raise ValueError(f"Operator {op_type} not supported")
            left = _evaluate(node.left)
            right = _evaluate(node.right)
            return _binops[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _unops:
                raise ValueError(f"Unary op {op_type} not supported")
            operand = _evaluate(node.operand)
            return _unops[op_type](operand)

        if isinstance(node, ast.Paren):
            return _evaluate(node.value)

        raise ValueError(f"Unsupported AST node {type(node)}")

    return _evaluate(node)


class QuadraticFieldElement:
    def __init__(self, u, v, c, n):
        self.u = u % n
        self.v = v % n
        self.c = c
        self.n = n

    def __repr__(self):
        return f"{self.u} + {self.v}√{self.c} mod {self.n}"

    def __add__(self, other):
        if not isinstance(other, QuadraticFieldElement):
            other = QuadraticFieldElement(other, 0, self.c, self.n)
        assert self.c == other.c and self.n == other.n
        return QuadraticFieldElement(self.u + other.u, self.v + other.v, self.c, self.n)

    def __sub__(self, other):
        if not isinstance(other, QuadraticFieldElement):
            other = QuadraticFieldElement(other, 0, self.c, self.n)
        return self + (-other)
    
    def __neg__(self):
        return QuadraticFieldElement(-self.u, -self.v, self.c, self.n)

    def __mul__(self, other):
        if not isinstance(other, QuadraticFieldElement):
            other = QuadraticFieldElement(other, 0, self.c, self.n)
        assert self.c == other.c and self.n == other.n
        u = (self.u * other.u + self.v * other.v * self.c) % self.n
        v = (self.u * other.v + self.v * other.u) % self.n
        return QuadraticFieldElement(u, v, self.c, self.n)

    def __rmul__(self, scalar):
        return QuadraticFieldElement(self.u * scalar, self.v * scalar, self.c, self.n)

    def __eq__(self, other):
        return (isinstance(other, QuadraticFieldElement)
                and self.u == other.u
                and self.v == other.v
                and self.c == other.c
                and self.n == other.n)

def oddprime(n):
    p = 3
    while True:
        if kronecker_symbol(p, n) == -1:
            return p
        p = nextprime(p)

def sqrtc(c, n):
    return QuadraticFieldElement(0, 1, c, n)

def xmat(n, a):
    two_a = a + a
    neg1  = -1 if not isinstance(a, QuadraticFieldElement) else QuadraticFieldElement(-1, 0, a.c, a.n)
    one   =  1 if not isinstance(a, QuadraticFieldElement) else QuadraticFieldElement(1, 0, a.c, a.n)
    zero  =  0 if not isinstance(a, QuadraticFieldElement) else QuadraticFieldElement(0, 0, a.c, a.n)
    return [[two_a, neg1], [one, zero]]

def mat_mul(A, B):
    if isinstance(B[0], list):
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
        ]
    return [A[0][0]*B[0] + A[0][1]*B[1], A[1][0]*B[0] + A[1][1]*B[1]]

def mat_pow(M, exponent):
    result = [[1, 0], [0, 1]]
    if isinstance(M[0][0], QuadraticFieldElement):
        one = QuadraticFieldElement(1, 0, M[0][0].c, M[0][0].n)
        zero = QuadraticFieldElement(0, 0, M[0][0].c, M[0][0].n)
        result = [[one, zero], [zero, one]]

    base = M
    e = exponent
    while e > 0:
        if e & 1:
            result = mat_mul(base, result)
        base = mat_mul(base, base)
        e >>= 1
    return result

def myisprime(n, a):
    M   = xmat(n, a)
    Mn  = mat_pow(M, n)
    one = 1 if not isinstance(a, QuadraticFieldElement) else QuadraticFieldElement(1, 0, a.c, a.n)
    xp  = mat_mul(Mn, [a, one])
    c      = a.c if isinstance(a, QuadraticFieldElement) else oddprime(n)
    target = QuadraticFieldElement(1, -1, c, n)
    return xp[1] == target

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if is_square(n):
        return False
    c = oddprime(n)
    a = QuadraticFieldElement(1, 1, c, n)
    return myisprime(n, a)

# --- End of code from cheby.py ---


# --- Flask Application ---

app = Flask(__name__)

# The HTML for the user interface is stored in this string.
# It uses Tailwind CSS for styling.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Primality Test</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
        <h1 class="text-2xl font-bold text-center text-gray-800 mb-2">Primality Tester</h1>
        <p class="text-center text-gray-500 mb-6">
            Enter a number or an expression. You can use operators: +, -, *, /, ^, (, ).
        </p>
        
        <form action="/" method="post" class="space-y-4">
            <div>
                <label for="number" class="block text-sm font-medium text-gray-700">Number to Test</label>
                <input type="text" name="number" id="number" 
                       class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                       placeholder="e.g., 29 or (5*6)-1"
                       value="{{ last_input or '' }}">
            </div>
            
            <div class="flex items-center space-x-4">
                <button type="submit"
                        class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    Test Number
                </button>
                <a href="/"
                   class="w-full flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    Clear
                </a>
            </div>
        </form>
        
        {% if result %}
        <div class="mt-6">
            <label for="result" class="block text-sm font-medium text-gray-700">Result</label>
            <div id="result" 
                   class="mt-1 p-3 w-full bg-gray-50 border border-gray-200 rounded-md text-gray-800">
                {{ result }}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result_text = ''
    last_input = ''
    if request.method == 'POST':
        expression = request.form['number']
        last_input = expression
        try:
            if not expression:
                raise ValueError("Input cannot be empty.")
            n = parse_int_expr(expression)
            if is_prime(n):
                result_text = f"It is prime."
            else:
                result_text = f"It is composite."
        except Exception as e:
            result_text = f"Error: {e}"
            
    return render_template_string(HTML_TEMPLATE, result=result_text, last_input=last_input)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
