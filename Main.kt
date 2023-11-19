// Copyright 2023 by Arkadiusz Bulski <arek.bulski@gmail.com> under MIT License

// This library does one thing and one thing only: it differentiates arbitrary mathematical functions.
// Well, that and it plots them and optimizes them and visualizes them and... yeah, just that.

// What it cannot do (yet):
// - integrate symbolically
// - function optimization is 90% baked

// Installing dependencies manually:
//     $ sudo apt install kotlin graphviz python3 python3-matplotlib firefox
// Compiling and running the code manually:
//     $ kotlinc Main.kt -include-runtime -d Main.jar
//     $ java -jar Main.jar

// ----------------------------------------------------------------------------
//                            Important imports (pun)

import java.io.File
import java.lang.IllegalStateException
import java.lang.StrictMath.*


// ----------------------------------------------------------------------------
//                 Core functionality of functions (also a pun)

typealias EvaluateFunc = (Double) -> Double
typealias DeriveFunc = () -> Function
typealias DescribeFunc = (String) -> String
typealias ChildrenFunc = () -> List<Function>
typealias LabelFunc = () -> String
typealias ReEvaluateFunc = (List<Function>) -> ((Double) -> Double)
typealias ReDeriveFunc = (List<Function>) -> (() -> Function)
typealias ToConstFunc = () -> Double?
typealias ToInfoFunc = () -> OptimizationInfo?
typealias PossiblyOptimizedFunc = () -> Function?

/**
 * Used internally to give functions and function graphs unique IDs. It helps to sort out what is what when debugging.
 */
private var nextID: Int = 1

/**
 * Used internally to pass information from one function to another what is its structure ie. its operation and
 * operands. This is used by phase 2 of the optimization algorithm. The problem it solves is that Function class
 * contains only lambdas and has no derived classes, and therefore its impossible to know what operation it represents
 * or to get its captured operands.
*/
data class OptimizationInfo (
    val op: String,
    val leftFunc: Function,
    val leftConst: Double,
    val rightFunc: Function,
    val rightConst: Double,
)

/**
 * Used internally.
 */
fun withEnglishSuffix (n: Int): String =
    n.toString() + when (n) {
        0 -> "th"
        1 -> "st"
        2 -> "nd"
        3 -> "rd"
        else -> "th"
    }

/**
 *    The Function class serves the core functionality (pardon the pun) of the library. Its instance represents an
 *    abstract mathematical function, and by means of the 10 lambdas it allows you to evaluate the function for some
 *    value x, differentiate it many many times, integrate it numerically, optimize it, visualize it with a graph, etc.
 *
 *    Note to users:
 *        You can evaluate it for a given x, using invoke operator:
 *            | val f = Sin.Of(X.PowerOf(2))
 *            | f(3.14159) returns -0.43028616647684903
 *            (Please do not use the Eval lambda, nor EvalCached method. Those are used internally.)
 *        You can differentiate it, using derive* methods:
 *            | f.derive() return a Function, f's derivative
 *            | f.deriveNth(5) returns a Function, f's fifth derivative
 *            (...which you can manipulate in all the same ways as the original function f.)
 *        You can integrate it numerically, using integrate method:
 *            | f.integrate(0.0, 10.0, 1000) returns a Double
 *        You can optimize it:
 *            | f.optimized() returns a Function
 *            (Look at FunctionGraph method optimized for details.)
 *        You can display its textual representation:
 *            | f.Description returns a String
 *            (Please do not use the Describe lambda, which is used internally.)
 *        You can display its graphical representation:
 *            (Look at FunctionGraph method exportedSvgs for details.)
 *
 *    Note to implementers:
 *        To make your own Function, you need to provide its primary constructor any or all 10 lambdas. Those are:
 *            Eval (x:Double -> Double) -- It evaluates the function for the given x. You do not need to be concerned
 *                  with caching, as that is taken care of. This is needed if you want to plot the function, integrate
 *                  it, or just to evaluate it for whatever reason.
 *            Derive (() -> Function) -- It differentiates the current function, and returns it as a Function instance.
 *                  This is needed if you want to obtain any of its derivatives, which is the whole point of using this
 *                  library btw.
 *            Describe (d:String -> String) -- It computes a textual representation of the function. You can embed
 *                  Description of the operands into it. The d argument is almost always "X", unless you start fiddling
 *                  with nesting functions in which case d will be a description of the nested function. This is needed
 *                  if you want (1) to see its textual mathematical form (2) to export it to SVG (3) to optimize it.
 *            Children (() -> List<Function>) -- It returns a list of Function instances that are its operands. This is
 *                  needed if you want (1) to export it to SVG (2) to optimize it.
 *            Label (() -> String) -- It returns a very short description of the function ie. "Sin" "X" "+" "2 *" etc.
 *                  This is needed when exporting the function into SVG format. The label is the text displayed in the
 *                  node, and that it all it does.
 *            ReEvaluate ((mc:List<Function>) -> ((x:Double) -> Double)) -- It allows you to rebuild a new class instance
 *                  that does the same thing as the original instance, but with different operands. The problem it
 *                  solves is that Function class is immutable. This lambda *should* return the exact same thing as
 *                  Eval lambda, except that you replace operands eg. this with mc[0], mc[1], etc. in same order as they
 *                  appear in Children lambda. This is needed if you want to optimize the function. All 3 phases of the
 *                  optimizing algorithm use this lambda.
 *            ReDerive ((mc:List<Function>) -> (() -> Function)) -- It allows you to rebuild a new class instance that
 *                  does the same thing as the original instance, but with different operands. The problem it solves is
 *                  that Function class is immutable. This lambda *should* return the exact same thing as Derive lambda,
 *                  except that you replace operands eg. this with mc[0], mc[1], etc. in same order as they appear in
 *                  Children lambda. This is needed if you want to optimize the function. All 3 phases of the optimizing
 *                  algorithm use this lambda.
 *            ToConst (() -> Double?) -- If some or all the operands are constants and therefore the current function is
 *                  itself a constant, this lambda returns that constant's value. This is optional, but useful if you
 *                  want to optimize the function. The phase 1 and 2 of the optimizing algorithm uses this lambda. If
 *                  unsure, just make it return a null.
 *            ToInfo (() -> OptimizationInfo?) -- It provides other functions with information what this instance does
 *                  and what are its operands. This information is necessary for some of the mathematical optimizations.
 *                  This is optional, but useful if you want to optimize the function. The phase 2 of the optimizing
 *                  algorithm uses this lambda. If unsure, just make it return a null.
 *            PossiblyOptimized (() -> Function?) -- It provides the optimization algorithm with a new, better instance
 *                  than the original instance. Usually this lambda refers to its operands ToConst and ToInfo lambdas.
 *                  This is optional, but useful if you want to optimize the function. The phase 2 of the optimizing
 *                  algorithm uses this lambda. If unsure, just make it return a null.
 */
class Function (
    val Eval: EvaluateFunc = { throw NotImplementedError("Eval lambda was not defined.") },
    val Derive: DeriveFunc = { throw NotImplementedError("Derive lambda was not defined.") },
    val Describe: DescribeFunc = { throw NotImplementedError("Describe lambda was not defined.") },
    val Children: ChildrenFunc = { throw NotImplementedError("Children lambda was not defined.") },
    val Label: LabelFunc = { throw NotImplementedError("Label lambda was not defined.") },
    val ReEvaluate: ReEvaluateFunc = { throw NotImplementedError("ReEvaluate lambda was not defined.") },
    val ReDerive: ReDeriveFunc = { throw NotImplementedError("ReDerive lambda was not defined.") },
    val ToConst: ToConstFunc = {null},
    val ToInfo: ToInfoFunc = {null},
    val PossiblyOptimized: PossiblyOptimizedFunc = {null},
) {

    /**
     * Secondary constructor, a minimalistic version of it. You can only use this function to evaluate and derive, no
     * optimization or exporting or anything fancy.
     */
    constructor (
        Eval: EvaluateFunc,
        Derive: DeriveFunc,
    ) : this (
        Eval,
        Derive,
        { throw NotImplementedError("Describe lambda was not defined.") },
        { throw NotImplementedError("Children lambda was not defined.") },
        { throw NotImplementedError("Label lambda was not defined.") },
        { throw NotImplementedError("ReEvaluate lambda was not defined.") },
        { throw NotImplementedError("ReDerive lambda was not defined.") },
        {null},
        {null},
        {null},
    )

    /**
     * Secondary constructor, a minimalistic version of it. You can only use this function to evaluate and derive, no
     * optimization or exporting or anything fancy.
     */
    constructor (
        Children: ChildrenFunc,
        ReEvaluate: ReEvaluateFunc,
        ReDerive: ReDeriveFunc,
    ) : this (
        ReEvaluate(Children()),
        ReDerive(Children()),
        { throw NotImplementedError("Describe lambda was not defined.") },
        Children,
        { throw NotImplementedError("Label lambda was not defined.") },
        ReEvaluate,
        ReDerive,
        {null},
        {null},
        {null},
    )

    /**
     * Secondary constructor, used internally to rebuild functions which are immutable.
     */
    constructor (originalFunc: Function, mutableChildren: List<Function>) : this(
        originalFunc.ReEvaluate(mutableChildren),
        originalFunc.ReDerive(mutableChildren),
        originalFunc.Describe,
        {mutableChildren.toList()},
        originalFunc.Label,
        originalFunc.ReEvaluate,
        originalFunc.ReDerive,
        originalFunc.ToConst,
        originalFunc.ToInfo,
        originalFunc.PossiblyOptimized,
    )

    /**
     * Secondary constructor, used internally to rebuild functions which are immutable.
     */
    fun possiblyRebuild (mutableChildren: List<Function>): Function{
        val thisChildren = this.Children()
        return if (thisChildren.indices.all({ i -> thisChildren[i] === mutableChildren[i] }))
            this
        else
            Function(this, mutableChildren)
    }

    /**
     * Used internally when exporting and useful during debugging.
     */
    val ID: Int = nextID++

    /**
     * Textual mathematical representation of the function. This is useful for public display, when debugging, but it
     * is also used internally when exporting graphs, optimizing, etc.
     */
    val Description: String = this.Describe("X")

    /**
     * Used internally by EvalCached.
     */
    private var CachedEvalX: Double = Double.NaN

    /**
     * Used internally by EvalCached.
     */
    private var CachedEvalResult: Double = Double.NaN

    /**
     * Used internally by invoke operator. Use the invoke operator instead.
     */
    fun EvalCached (x: Double): Double {
        if (x == CachedEvalX) {
            return CachedEvalResult
        } else {
            CachedEvalX = x
            CachedEvalResult = Eval(x)
            return CachedEvalResult
        }
    }

    /**
     * Returns the next derivative. Please use this instead of Derive lambda.
     */
    fun derive (): Function =
        this.Derive()

    /**
     * Returns a derivative of a derivative of a derivative... of the function.
     */
    fun deriveNth (n: Int): Function {
        require(n >= 0) { "You can only demand 0-th or higher derivative." }
        return (1..n).fold(this, { f,i -> f.Derive() })
    }

    /**
     * Returns several successive derivatives in a list.
     */
    fun deriveMany (n: Int): List<Function> {
        require(n >= 0) { "You can only demand 0 or more derivatives." }
        return (1..n).runningFold(this, { f,i -> f.Derive() })
    }

    /**
     * Returns an optimized version of the function. Its a wrapper around FunctionGraph.optimized method.
     */
    fun optimized (): Function =
        FunctionGraph(this,0).optimized().DerivativeFunctions[0]

    /**
     * Computes a definite integral over the function numerically.
     */
    fun integrate (a: Double, b: Double, count: Int): Double {
        require(count >= 1, {"Amount of points (count) must be at least 1."})
        var sum = 0.0
        val width = (b-a)/count
        for (i in 1..count) {
            sum += this(a + width*i)
        }
        return sum * width
    }

    override fun toString(): String =
        "Function ID=$ID is $Description"
    override fun equals(other: Any?): Boolean =
        (other is Function) && (Description == other.Description)
    override fun hashCode(): Int =
        Description.hashCode()
}

/**
 * Function evaluation operator. Use it instead of Eval lambda and EvalCached method.
 */
operator fun Function.invoke (other: Double): Double =
    this.EvalCached(other)

/**
 * General purpose operator for +f. Yes, it is completely useless. Its only purpose is to make optimization harder.
 */
operator fun Function.unaryPlus (): Function =
    Function(
        {x -> this(x)},
        {+this.Derive()},
        {d -> this.Describe(d) },
        {listOf(this)},
        {"unary +"},
        {mc -> {x -> mc[0](x)}},
        {mc -> {mc[0].Derive()}},
        // Optimizes +(a) into (a) constant.
        {
            return@Function this.ToConst()
        },
        {OptimizationInfo("+f", this, Double.NaN, this, Double.NaN)},
        // Optimizes +(f) into (f) function.
        {
            return@Function this
        },
    )

/**
 * General purpose operator for -f.
 */
operator fun Function.unaryMinus (): Function =
    Function(
        {x -> -this(x)},
        {-this.Derive()},
        {d -> "-(${this.Describe(d)})"},
        {listOf(this)},
        {"unary -"},
        {mc -> {x -> -mc[0](x)}},
        {mc -> {-mc[0].Derive()}},
        // Optimizes -(a) into (-a) constant.
        {
            val right = this.ToConst()
            if (right != null)
                return@Function -right
            null
        },
        {OptimizationInfo("-f", this, Double.NaN,this, Double.NaN)},
        // Optimizes -(-(f)) into (f) function.
        {
            val right = this.ToInfo()
            if (right != null && right.op == "-f")
                return@Function right.rightFunc
            null
        },
    )

/**
 *  Syntactic sugar for (a+f) without using Value.
 *  Preserves (a+f) optimization, as opposed to (f+a).
 */
operator fun Double.plus (other: Function) =
    Value(this)+other

/**
 *  Syntactic sugar for (f+a) without using Value.
 *  Optimizes (f+a) into (a+f), which is useful down the line.
 */
operator fun Function.plus (other: Double) =
    Value(other)+this

/**
 * General purpose operator for f+g.
 */
operator fun Function.plus (other: Function): Function =
    Function(
        {x -> this(x)+other(x)},
        {this.Derive()+other.Derive()},
        {d -> "(${this.Describe(d)}) + (${other.Describe(d)})"},
        {listOf(this, other)},
        {"+"},
        {mc -> {x -> mc[0](x)+mc[1](x)}},
        {mc -> {mc[0].Derive()+mc[1].Derive()}},
        // Optimizes (a)+(b) into (a+b) constant.
        {
            val left = this.ToConst()
            val right = other.ToConst()
            if (left != null && right != null)
                return@Function left+right
            null
        },
        {OptimizationInfo("f+g", this, Double.NaN, other, Double.NaN)},
        // Optimizes (0+f) and (f+0) into (f) function.
        // Optimizes a+(b+f) into (a+b)+f function.
        // Optimizes a+(f+b) into (a+b)+f function.
        {
            val left = this.ToInfo()
            val right = other.ToInfo()
            if (left != null && left.op == "a" && left.leftConst == 0.0)
                return@Function other
            if (right != null && right.op == "a" && right.leftConst == 0.0)
                return@Function this
            if (left != null && left.op == "a" && right != null && right.op == "f+g" && right.leftFunc.ToConst() != null)
                return@Function (left.leftConst + right.leftFunc.ToConst()!!) + right.rightFunc
            if (left != null && left.op == "a" && right != null && right.op == "f+g" && right.rightFunc.ToConst() != null)
                return@Function (left.leftConst + right.rightFunc.ToConst()!!) + right.leftFunc
            null
        },
    )

/**
 *  Syntactic sugar for (a-f) without using Value.
 *  Preserves (a-f) optimization, as opposed to (f-a).
 */
operator fun Double.minus (other: Function) =
    Value(this)-other

/**
 *  Syntactic sugar for (f-a) without using Value.
 */
operator fun Function.minus (other: Double) =
    this-Value(other)

/**
 *  General purpose operator for f-g.
 */
operator fun Function.minus (other: Function): Function =
    Function(
        {x -> this(x)-other(x)},
        {this.Derive()-other.Derive()},
        {d -> "(${this.Describe(d)}) - (${other.Describe(d)})"},
        {listOf(this, other)},
        {"-"},
        {mc -> {x -> mc[0](x)-mc[1](x)}},
        {mc -> {mc[0].Derive()-mc[1].Derive()}},
        // Optimizes (a)-(b) into (a-b) constant.
        {
            val left = this.ToConst()
            val right = other.ToConst()
            if (left != null && right != null)
                return@Function left-right
            null
        },
        {OptimizationInfo("f-g", this, Double.NaN, other, Double.NaN)},
        // Optimizes (0-f) into (-f) function.
        // Optimizes (f-0) into (f) function.
        // Optimizes a-(b-(f)) into ((a-b)+f) function.
        {
            val left = this.ToInfo()
            val right = other.ToInfo()
            if (left != null && left.op == "a" && left.leftConst == 0.0)
                return@Function -other
            if (right != null && right.op == "a" && right.leftConst == 0.0)
                return@Function this
            if (left != null && left.op == "a" && right != null && right.op == "f-g" && right.leftFunc.ToConst() != null)
                return@Function (left.leftConst - right.leftFunc.ToConst()!!) + right.rightFunc
            null
        },
    )

/**
 *  General purpose operator for a*f.
 */
operator fun Double.times (other: Function): Function =
    Function(
        {x -> this*other(x)},
        {this*(other.Derive())},
        {d -> "($this) * (${other.Describe(d)})"},
        {listOf(other)},
        {"$this *"},
        {mc -> {x -> this*mc[0](x)}},
        {mc -> {this*(mc[0].Derive())}},
        // Optimizes (0*f) and (f*0) into 0 constant.
        // Optimizes (a)*(b) into (a*b) constant.
        {
            val right = other.ToConst()
            if (this == 0.0 || right == 0.0)
                return@Function 0.0
            if (right != null)
                return@Function this*right
            null
        },
        {OptimizationInfo("a*f", Value(this), this, other, Double.NaN)},
        // Optimizes (1*f) into (f) function.
        // Optimizes a*(b*f) into (a*b)*f function.
        {
            val right = other.ToInfo()
            if (this == 1.0)
                return@Function other
            if (right != null && right.op == "a*f")
                return@Function (this * right.leftConst) * right.rightFunc
            null
        },
    )

/**
 *  Syntactic sugar for (f*a) without using Value.
 *  Preserves (a*f) optimization, as opposed to (f*a).
 */
operator fun Function.times (other: Double) =
    other*this

/**
 *  General purpose operator for f*g.
 */
operator fun Function.times (other: Function): Function =
    Function(
        {x -> this(x)*other(x)},
        {this.Derive()*other+this*other.Derive()},
        {d -> "(${this.Describe(d)}) * (${other.Describe(d)})"},
        {listOf(this,other)},
        {"*"},
        {mc -> {x -> mc[0](x)*mc[1](x)}},
        {mc -> {mc[0].Derive()*mc[1]+mc[0]*mc[1].Derive()}},
        // Optimizes (0*f) and (f*0) into 0 constant.
        // Optimizes (a)*(b) into (a*b) constant.
        {
            val left = this.ToConst()
            val right = other.ToConst()
            if (left == 0.0 || right == 0.0)
                return@Function 0.0
            if (left != null && right != null)
                return@Function left*right
            null
        },
        {
            val left = this.ToInfo()
            val right = other.ToInfo()
            if (left?.op == "a")
                return@Function OptimizationInfo("a*f", Value(left.leftConst), left.leftConst, other, Double.NaN)
            if (right?.op == "a")
                return@Function OptimizationInfo("a*f", Value(right.leftConst), right.leftConst, this, Double.NaN)
            OptimizationInfo("f*g", this, Double.NaN, other, Double.NaN)
        },
        // Optimizes (a*f) and (f*a) into a*(f) function using Double.times(Function) operator.
        {
            val left = this.ToInfo()
            val right = other.ToInfo()
            if (left != null && left.op == "a")
                return@Function left.leftConst * other
            if (right != null && right.op == "a")
                return@Function right.leftConst * this
            null
        },
    )

/**
 *  General purpose operator for a/f.
 */
operator fun Double.div (other: Function): Function =
    Function(
        {x -> this/other(x)},
        {this/(other.Derive().Of(1.0/other))},
        {d -> "($this) / (${other.Describe(d)})"},
        {listOf(other)},
        {"$this / "},
        {mc -> {x -> this/mc[0](x)}},
        {mc -> {this/(mc[0].Derive().Of(1.0/mc[0]))}},
        // Optimizes (0/f) into 0 constant.
        // Optimizes (a/b) into (a/b) constant.
        {
            val right = other.ToConst()
            if (this == 0.0)
                return@Function 0.0
            if (right != null)
                return@Function this/right
            null
        },
        // No optimization is achieved with this info.
        {OptimizationInfo("a/f", Value(this), this, other, Double.NaN)},
        // No optimization is achieved here.
        {null},
    )

/**
 *  General purpose operator for f/g.
 */
operator fun Function.div (other: Function): Function =
    Function(
        {x -> this(x)/other(x)},
        {(this.Derive()*other-this*other.Derive())/(other*other)},
        {d -> "(${this.Describe(d)}) / (${other.Describe(d)})"},
        {listOf(this,other)},
        {"*"},
        {mc -> {x -> mc[0](x)/mc[1](x)}},
        {mc -> {(mc[0].Derive()*mc[1]-mc[0]*mc[1].Derive())/(mc[1]*mc[1])}},
        // Optimizes (0/f) into 0 constant.
        // Optimizes (a/b) into (a/b) constant.
        {
            val left = this.ToConst()
            val right = other.ToConst()
            if (left == 0.0)
                return@Function 0.0
            if (left != null && right != null)
                return@Function left/right
            null
        },
        {OptimizationInfo("f/g", this, Double.NaN, other, Double.NaN)},
        // Optimizes (f/a) into (1/a)*(f) function using Double.times(Function) operator.
        {
            val right = other.ToInfo()
            if (right != null && right.op == "a")
                return@Function (1.0/right.leftConst)*this
            null
        },
    )

/**
 *  General purpose singleton for the X variable. Need I say more?
 */
val X: Function =
    Function(
        {x -> x},
        {Value(1.0)},
        {d -> d},
        {listOf()},
        {"X"},
        {mc -> {x -> x}},
        {mc -> {Value(1.0)}},
        // No optimization is achieved here.
        {null},
        // No OptimizationInfo is provided because it would have to be self-referential.
        {null},
        // No optimization is achieved here.
        {null},
    )

/**
 *  General purpose function for creating constant values. Note that there are many sugary operators that do not require
 *  you to use a Value eg. 5.0*Sin will work the same as Value(5.0)*Sin.
 */
fun Value (value: Double): Function =
    Function(
        {x -> value},
        {Value(0.0)},
        {"$value"},
        {listOf()},
        {"$value"},
        {mc -> {x -> value}},
        {mc -> {Value(0.0)}},
        {value},
        // This OptimizationInfo is used throughout entire codebase to detect special conditions eg. (a*f) instead of more general (f*g).
        {OptimizationInfo("a", Value(value), value, Value(value), value)},
        {null},
    )

/**
 *  Function for creating the (f ** n) power function, given both the base function and integer exponent.
 */
fun Function.PowerOf (exponent: Int): Function =
    Function(
        {x -> pow(this(x),exponent.toDouble())},
        {(exponent.toDouble())*(this.PowerOf(exponent-1))*(this.Derive())},
        {d -> "(${this.Describe(d)}) ** (${exponent.toDouble()})"},
        {listOf(this)},
        {" ** ${exponent}"},
        {mc -> {x -> pow(mc[0](x),exponent.toDouble())}},
        {mc -> {(exponent.toDouble())*(mc[0].PowerOf(exponent-1))*(mc[0].Derive())}},
        // Optimizes (a**b) into (a**b) constant.
        // Optimizes (a**0) into 1 constant.
        {
            val left = this.ToConst()
            if (left != null)
                return@Function pow(left,exponent.toDouble())
            if (exponent == 0)
                return@Function 1.0
            null
        },
        {null},
        // Optimizes (f**1) into (f).
        // Optimizes (f**2) into (f*f).
        {
            if (exponent == 1)
                return@Function this
            if (exponent == 2)
                return@Function this*this
            null
        },
    )

/**
 *  Function for creating the (a ** f) exponential function, given both the floating-point base and exponent function.
 */
fun Double.ExpOf (exponent: Function): Function =
    Function(
        {x -> pow(this,exponent(x))},
        {(this.ExpOf(exponent))*(Log.Of(Value(this)))*(exponent.Derive())},
        {d -> "($this) ** (${exponent.Describe(d)})"},
        {listOf(exponent)},
        {"$this ** "},
        {mc -> {x -> pow(this,mc[1](x))}},
        {mc -> {(this.ExpOf(mc[0]))*(Log.Of(Value(this)))*(mc[0].Derive())}},
        // Optimizes (0**f) into 0 constant.
        // Optimizes (1**f) into 1 constant.
        // Optimizes (a**b) into (a**b) constant.
        {
            val right = exponent.ToConst()
            if (this == 0.0)
                return@Function 0.0
            if (this == 1.0)
                return@Function 1.0
            if (right != null)
                return@Function pow(this,right)
            null
        },
        {null},
        // No optimization is achieved here.
        {null},
    )

/**
 * Singleton for natural logarithm function.
 */
val Log: Function =
    Function(
        {x -> log(x)},
        {1.0/X},
        {d -> "Log($d)"},
        {listOf(X)},
        {"Log"},
        {mc -> {x -> log(x)}},
        {mc -> {1.0/mc[0]}},
        // No optimization is achieved here.
        {null},
        {null},
        // No optimization is achieved here.
        {null},
    )

/**
 * Singleton for the sine trigonometric function.
 */
val Sin: Function =
    Function(
        {x -> sin(x)},
        {Cos},
        {d -> "Sin($d)"},
        {listOf(X)},
        {"Sin"},
        {mc -> {x -> sin(x)}},
        {mc -> {Cos}},
        // No optimization is achieved here.
        {null},
        {null},
        // No optimization is achieved here.
        {null},
    )

/**
 * Singleton for the cosine trigonometric function.
 */
val Cos: Function =
    Function(
        {x -> cos(x)},
        {-Sin},
        {d -> "Cos($d)"},
        {listOf(X)},
        {"Cos"},
        {mc -> {x -> cos(x)}},
        {mc -> {-Sin}},
        // No optimization is achieved here.
        {null},
        {null},
        // No optimization is achieved here.
        {null},
    )

/**
 * The most impressive operator of them all, the nesting method.
 */
fun Function.Of (other: Function): Function =
    Function(
        {x -> this(other(x))},
        {this.Derive().Of(other)*(other.Derive())},
        {d -> this.Describe(other.Describe(d)) },
        {listOf(this,other)},
        {"of"},
        {mc -> {x -> mc[0](mc[1](x))}},
        {mc -> {mc[0].Derive().Of(mc[1])*(mc[1].Derive())}},
        // Optimizes (a.Of(f)) into (a) constant. Rarely needed.
        // Optimizes (f.Of(b)) into (f.Of(b)) constant, regardless of f.
        {
            val left = this.ToConst()
            val right = other.ToConst()
            if (left != null)
                return@Function left
            if (right != null)
                return@Function this(right)
            null
        },
        {null},
        // No optimization is achieved here.
        {null},
    )


// ----------------------------------------------------------------------------
//                     Core functionality of function graphs

/**
 * The FunctionGraph class represents both an original function and several of its derivatives. The difference between
 * a FunctionGraph and a list of Functions is that a graph can show you overlapping subfunctions.
 *
 * Note to users:
 *     You can access the derivatives:
 *         | val fg = FunctionGraph(Sin, 4)
 *         | fg.DerivativeFunctions[0] returns a Function, the original function.
 *         | fg.DerivativeFunctions[1] returns a Function, the first derivative.
 *     You can optimize the functions:
 *         | fg.optimized() returns a new FunctionGraph
 *     You can export the functions graph of subfunctions into SVG:
 *         | fg.exportedGraphsSvgs("filename") returns itself
 *     You can export the functions combined plot into SVG:
 *         | fg.exportedPlotSvg("filename") returns itself
 */
class FunctionGraph (originalFunc: Function, derivatives: Int) {

    /**
     * Used to store the original function and several of its derivatives.
     */
    var DerivativeFunctions: MutableList<Function> = originalFunc.deriveMany(derivatives).toMutableList()

    /**
     * Used internally when exporting and useful during debugging.
     */
    val ID: Int = nextID++

    /**
     * Secondary constructor, used internally by the optimized method.
     */
    constructor (DerivativeFunctions: MutableList<Function>) : this(Value(Double.NaN), 0) {
        this.DerivativeFunctions = DerivativeFunctions
    }

    /**
     * Returns a new FunctionGraph that contains equivalent derivative functions, but with less nodes. The process is
     * both deterministic (same functions will always be optimized into same equivalent functions) and adaptive (the
     * optimizer will repeat its phases until they stop giving better results).
     */
    fun optimized(): FunctionGraph {
        var DerivativeFunctions = this.DerivativeFunctions.toMutableList()

        // Phase 1: full constant expressions turn into constants.
        for (derivative in DerivativeFunctions.indices) {
            fun traverse (func: Function): Function {
                var mutableChildren = func.Children().toMutableList()

                for (childIndex in mutableChildren.indices) {
                    mutableChildren[childIndex] = traverse(mutableChildren[childIndex])
                }

                val r = func.possiblyRebuild(mutableChildren)
                val c = r.ToConst()
                return if (c != null) Value(c) else r
            }
            DerivativeFunctions[derivative] = traverse(DerivativeFunctions[derivative])
        }

        while (true) {
            var OldDerivativeFunctions = DerivativeFunctions.toMutableList()

            // Phase 2: partial constant expressions turn into (better) functions.
            for (derivative in DerivativeFunctions.indices) {
                fun traverse (func: Function): Function {
                    var mutableChildren = func.Children().toMutableList()

                    for (childIndex in mutableChildren.indices) {
                        mutableChildren[childIndex] = traverse(mutableChildren[childIndex])
                    }

                    var r = func.possiblyRebuild(mutableChildren)
                    return r.PossiblyOptimized() ?: r
                }
                DerivativeFunctions[derivative] = traverse(DerivativeFunctions[derivative])
            }

            if (DerivativeFunctions.indices.all({ i -> DerivativeFunctions[i] === OldDerivativeFunctions[i] }))
                break
        }

        // Phase 3: deduplication of paths in the dag of subfunctions.
        var deduplicatedPaths = mutableMapOf<String,Function>()
        for (derivative in DerivativeFunctions.indices) {
            fun traverse (func: Function): Function {
                if (deduplicatedPaths.containsKey(func.Description)) {
                    return deduplicatedPaths[func.Description]!!
                } else {
                    deduplicatedPaths[func.Description] = func
                }

                var mutableChildren = func.Children().toMutableList()

                for (childIndex in mutableChildren.indices) {
                    mutableChildren[childIndex] = traverse(mutableChildren[childIndex])
                    val childDescription = mutableChildren[childIndex].Description
                    if (deduplicatedPaths.containsKey(childDescription)) {
                        mutableChildren[childIndex] = deduplicatedPaths[childDescription]!!
                    } else {
                        deduplicatedPaths[childDescription] = mutableChildren[childIndex]
                    }
                }

                return func.possiblyRebuild(mutableChildren)
            }
            DerivativeFunctions[derivative] = traverse(DerivativeFunctions[derivative])
        }

        return FunctionGraph(DerivativeFunctions)
    }

    /**
     * Exports the graph containing the functions into an SVG file, and optionally opens it in the firefox browser.
     * Returns itself so you can do chaining.
     */
    fun exportedGraphsSvgs (outputFilename: String, openInBrowser: Boolean = false): FunctionGraph {
        require(outputFilename.isNotEmpty()) { "Please provide a filename, anything." }
        require(!outputFilename.contains('.')) { "Please provide a filename (without extension) for .dot and .svg files." }

        val originalFuncDescription = DerivativeFunctions[0].Description
        var dotTasks = mutableListOf<Process>()

        for (derivative in 0..DerivativeFunctions.lastIndex) {
            // Phase 1
            var deduplicatedNodes = hashMapOf<Int,Function>()
            var deduplicatedEdges = hashSetOf<Pair<Int,Int>>()
            var nodeDuplicateSubnodesCounts = hashMapOf<Int,Int>()

            fun traverseNodesAndEdges (func: Function): Int {
                deduplicatedNodes[func.ID] = func
                var duplicateSubnodes = func.Children().size
                for (child in func.Children()) {
                    deduplicatedEdges.add(Pair(func.ID, child.ID))
                    duplicateSubnodes += traverseNodesAndEdges(child)
                }
                nodeDuplicateSubnodesCounts[func.ID] = duplicateSubnodes
                return duplicateSubnodes
            }
            for (subderivative in 0..derivative) {
                traverseNodesAndEdges(DerivativeFunctions[subderivative])
            }

            // Phase 2
            var nodeUniqueSubnodeSets = hashMapOf<Int,HashSet<Int>>()

            fun traverseUnique (uniqueNodes: HashSet<Int>, func: Function) {
                if (uniqueNodes.add(func.ID)) {
                    for (child in func.Children()) {
                        traverseUnique(uniqueNodes, child)
                    }
                }
            }
            for (subderivative in 0..derivative) {
                val root = DerivativeFunctions[subderivative]
                var uniqueNodes = hashSetOf<Int>()
                nodeUniqueSubnodeSets[root.ID] = uniqueNodes
                traverseUnique(uniqueNodes, root)
            }

            // Phase 3
            var sb = StringBuilder()
            sb.append("""
            digraph {
                graph [
                    layout = dot
                    tooltip = "Hey! Notice that red nodes are clickable and \n tooltips contain additional information!"
                ]
                node [
                    style = filled
                ]
            """)

            for (subderivative in 0..derivative) {
                val subderivativefid = DerivativeFunctions[subderivative].ID
                sb.append("""
                function${subderivative} [
                    label = "${if (subderivative == 0) "$originalFuncDescription \n original function" else "$originalFuncDescription \n ${withEnglishSuffix(subderivative)} derivative"}"
                    tooltip = "${if (subderivative == 0) "original function" else "${withEnglishSuffix(subderivative)} derivative function"} \n contains ${nodeDuplicateSubnodesCounts[subderivativefid]!!+1} duplicate subfunctions \n contains ${nodeUniqueSubnodeSets[subderivativefid]!!.size} unique subfunctions \n (and is clickable)"
                    URL = "${outputFilename}-${subderivative}.svg"
                    color = red
                    style = "rounded,filled"
                    shape = box
                ]
                function${subderivative} -> f${subderivativefid}
                """)
            }
            for (node in deduplicatedNodes.values) {
                sb.append("""
                f${node.ID} [
                    label="${node.Label()}"
                    tooltip="Function ID=${node.ID} \n contains ${nodeDuplicateSubnodesCounts[node.ID]} duplicate subfunctions"
                    color=${if (node.Children().isNotEmpty()) "lightblue" else "lightgreen"}
                ]
                """)
            }
            for (edge in deduplicatedEdges) {
                sb.append("""
                f${edge.first} -> f${edge.second} [
                    tooltip="Function ID=${edge.first} depends on function ID=${edge.second}"
                ]
                """)
            }
            sb.append("""
            } // end of digraph
            """)

            // Phase 4
            val output = sb.toString().replace("    ", "")
            val dotFilename = "${outputFilename}-${derivative}.dot"
            File(dotFilename).writeText(output)
            val svgFilename = "${outputFilename}-${derivative}.svg"
            var dotTask = ProcessBuilder("dot", "-Tsvg", dotFilename, "-o", svgFilename)
                .start()
            dotTasks.add(dotTask)
        }

        for (task in dotTasks) {
            if (task.waitFor() != 0)
                throw IllegalStateException("The dot (graphviz) command failed.")
        }

        if (openInBrowser) {
            val svgFilename = "${outputFilename}-${DerivativeFunctions.lastIndex}.svg"
            var firefoxTask = ProcessBuilder("firefox", svgFilename)
                .start()
            if (firefoxTask.waitFor() != 0)
                throw IllegalStateException("The firefox command failed.")
        }

        return this
    }

    /**
     * Exports a combined plot of the functions for a given range of Xs into an SVG file, and optionally opens it
     * in the firefox browser. Returns itself so you can do chaining.
     */
    fun exportedPlotSvg (outputFilename: String, minX: Double, maxX: Double, openInBrowser: Boolean = false): FunctionGraph {
        require(outputFilename.isNotEmpty()) { "Please provide a filename, anything." }
        require(!outputFilename.contains('.')) { "Please provide a filename (without extension) for .svg file." }
        val pyFilename = "${outputFilename}.py"
        val svgFilename = "${outputFilename}.svg"

        var sb = StringBuilder()
        sb.append("""
            from matplotlib import pyplot
            NaN = float("NaN")
            Infinity = float("inf")
            fig,ax = pyplot.subplots()
        """)

        for (derivative in DerivativeFunctions.indices) {
            val xs = (0..1000).map({i -> minX+i*(maxX-minX)/1000}).toList()
            val f = DerivativeFunctions[derivative]
            val ys = xs.map({x -> f(x)})
            sb.append("""
                xs=[${xs.joinToString()}]
                ys=[${ys.joinToString()}]
                ax.plot(xs, ys, label="${withEnglishSuffix(derivative)} derivative")
            """)
        }

        sb.append("""
            ax.set(title="${DerivativeFunctions[0].Description} up to ${withEnglishSuffix(DerivativeFunctions.lastIndex)} derivative")
            ax.legend()
            fig.savefig("${svgFilename}")
        """)

        val output = sb.toString().replace("    ", "")
        File(pyFilename).writeText(output)
        var pytask = ProcessBuilder("python3", pyFilename)
            .start()
        if (pytask.waitFor() != 0)
            throw IllegalStateException("The python3 (matplotlib) command failed.")

        if (openInBrowser) {
            var firefoxTask = ProcessBuilder("firefox", svgFilename)
                .start()
            if (firefoxTask.waitFor() != 0)
                throw IllegalStateException("The firefox command failed.")
        }

        return this
    }

    override fun toString(): String =
        "FunctionGraph ID=$ID of ${DerivativeFunctions[0].Description} and its ${DerivativeFunctions.lastIndex} derivatives"
    override fun equals(other: Any?): Boolean =
        (other is FunctionGraph) && (this.DerivativeFunctions == other.DerivativeFunctions)
    override fun hashCode(): Int =
        this.DerivativeFunctions.hashCode()
}

/**
 * Used for debugging and presentations.
 */
fun prepareForInspections() {
    var firefoxTask = ProcessBuilder("firefox", "--new-window")
        .start()
    if (firefoxTask.waitFor() != 0)
        throw IllegalStateException("The firefox command failed.")
}

/**
 * Used for debugging and presentations.
 */
fun inspect(func: Function, derivatives: Int) {
    val outputFilename = "inspected-f${func.ID}"
    FunctionGraph(func, derivatives)
        .exportedPlotSvg("${outputFilename}-plot", 0.0, 10.0, true)
        .exportedGraphsSvgs("${outputFilename}-graph", true)
        .optimized()
        .exportedGraphsSvgs("${outputFilename}-optimized", true)
}

fun main() {
    prepareForInspections()
    inspect(Sin.PowerOf(5), 3)
    inspect((2.0.ExpOf(Sin)), 3)
    inspect(X.PowerOf(5), 5)
    inspect(0.9.ExpOf(X), 2)
    inspect(X.PowerOf(5), 6)
    inspect(X.PowerOf(3)+100.0*Sin, 3)
}
