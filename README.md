# Swift矩阵与向量运算

抽时间写了一下swift上的向量与矩阵运算，使用的是原生的Accelerate框架，便于之后的开发。

*(注：欢迎随意使用源码，如需转载请注明出处)*

### 基础框架与类型定义

```swift
import Foundation
import Accelerate

typealias Matrix = Array<[Double]>
typealias Vector = [Double]
```

### 向量运算
基础运算：
1. 向量相加
2. 向量相减
3. 向量点乘常数

```swift
func vecAdd(vec1:Vector, vec2:Vector) -> Vector {
    var addresult = Vector(repeating: 0.0, count: vec1.count)
    vDSP_vaddD(vec1, 1, vec2, 1, &addresult, 1, vDSP_Length(vec1.count))
    return addresult
}

func vecSub(vec1:Vector, vec2:Vector) -> Vector {
    var subresult = Vector(repeating: 0.0, count: vec1.count)
    vDSP_vsubD(vec2, 1, vec1, 1, &subresult, 1, vDSP_Length(vec1.count))
    return subresult
}

func vecScale(vec:Vector, num:Double) -> Vector {
    var n = num
    var vsresult = Vector(repeating: 0.0, count: vec.count)
    vDSP_vsmulD(vec, 1, &n, &vsresult, 1, vDSP_Length(vec.count))
    return vsresult
}

vecScale(vec: [1, 3], num: 0.6)
```

均值化:
1. 均值向量
2. 均值化

```swift
// Mean Vector
func meanVector(inputMatrix:Matrix) -> Vector {
    let vecDimension = inputMatrix[0].count
    let vecCount = Double(inputMatrix.count)
    let sumVec = inputMatrix.reduce(Vector(repeating: 0.0, count: vecDimension),{vecAdd(vec1: $0, vec2: $1)})
    let averageVec = sumVec.map({$0/vecCount})
    return averageVec
}

// Mean Normalization
func meanNormalization(inputMatrix:Matrix) -> Matrix {
    let averageVec = meanVector(inputMatrix: inputMatrix)
    let outputMatrix = inputMatrix.map({vecSub(vec1: $0, vec2: averageVec)})
    return outputMatrix
}
```

### 矩阵运算
基础运算
1. 矩阵相加
2. 矩阵相减
3. 矩阵相乘
4. 矩阵点乘常数

```swift
func matAdd(mat1:Matrix, mat2:Matrix) -> Matrix {
    var outputMatrix:Matrix = []
    for i in 0..<mat1.count {
        let vec1 = mat1[i]
        let vec2 = mat2[i]
        outputMatrix.append(vecAdd(vec1: vec1, vec2: vec2))
    }
    return outputMatrix
}

func matSub(mat1:Matrix, mat2:Matrix) -> Matrix {
    var outputMatrix:Matrix = []
    for i in 0..<mat1.count {
        let vec1 = mat1[i]
        let vec2 = mat2[i]
        outputMatrix.append(vecSub(vec1: vec1, vec2: vec2))
    }
    return outputMatrix
}

func matMul(mat1:Matrix, mat2:Matrix) -> Matrix {
    let m = mat1[0].count
    let n = mat2.count
    let p = mat1.count
    var mulresult = Vector(repeating: 0.0, count: m*n)
    let mat1t = transpose(inputMatrix: mat1)
    let mat1vec = mat1t.reduce([], {$0+$1})
    let mat2t = transpose(inputMatrix: mat2)
    let mat2vec = mat2t.reduce([], {$0+$1})
    vDSP_mmulD(mat1vec, 1, mat2vec, 1, &mulresult, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(p))
    var outputMatrix:Matrix = []
    for i in 0..<n {
        outputMatrix.append(Array(mulresult[i*m..<i*m+m]))
    }
    return outputMatrix
}

func matScale(mat:Matrix, num:Double) -> Matrix {
    let outputMatrix = mat.map({vecScale(vec: $0, num: num)})
    return outputMatrix
}
```

高级运算
1. 矩阵转置transport
2. 矩阵求逆invert
3. 协方差矩阵Covariance Matrix

```swift
func transpose(inputMatrix: Matrix) -> Matrix {
    let m = inputMatrix[0].count
    let n = inputMatrix.count
    let t = inputMatrix.reduce([], {$0+$1})
    var result = Vector(repeating: 0.0, count: m*n)
    vDSP_mtransD(t, 1, &result, 1, vDSP_Length(m), vDSP_Length(n))
    var outputMatrix:Matrix = []
    for i in 0..<m {
        outputMatrix.append(Array(result[i*n..<i*n+n]))
    }
    return outputMatrix
}

func invert(inputMatrix: Matrix) -> Matrix? {
    
    var outputMatrix:Matrix = []
    
    let count = inputMatrix.count
    var inMatrix = inputMatrix.reduce([], {$0+$1})
    var N = __CLPK_integer(sqrt(Double(inMatrix.count)))
    var pivots = [__CLPK_integer](repeating: 0, count: Int(N))
    var workspace = 0.0
    var error : __CLPK_integer = 0
    
    dgetrf_(&N, &N, &inMatrix, &N, &pivots, &error)
    
    if error != 0 {
        for i in 0..<count {
            let s = i * count
            let sss = inMatrix[s...(s+count-1)]
            outputMatrix.append(Array(sss))
        }
        
        return outputMatrix
    }
    
    dgetri_(&N, &inMatrix, &N, &pivots, &workspace, &N, &error)
    
    for i in 0..<count {
        let s = i * count
        let sss = inMatrix[s...(s+count-1)]
        outputMatrix.append(Array(sss))
    }
    
    return outputMatrix
}

// Covariance Matrix
func covarianceMatrix(inputMatrix:Matrix) -> Matrix {
    let t = transpose(inputMatrix: inputMatrix)
    return matMul(mat1: inputMatrix, mat2: t)
}
```

部分测试

```swift
meanNormalization(inputMatrix: [[1.0,2.0,4.0],[3.0,3.0,5.0]])
matAdd(mat1: [[1.0, 2.0],[3.0, 3.0],[3.0, 2.0]], mat2: [[1.0, 2.0],[3.0, 3.0],[3.0, 100.0]])
matSub(mat1: [[1.0, 2.0],[3.0, 3.0],[3.0, 2.0]], mat2: [[1.0, 2.0],[3.0, 3.0],[4.0, 100.0]])
matMul(mat1: [[1.0],[3.0],[3.0]], mat2: [[1.0, 2.0, 4.0],[3.0,3.0, 6.0]])
matScale(mat: [[1.0],[3.0],[3.0]], num: 0.6)
transpose(inputMatrix: [[1.0,2.0,7.0],[3.0,4.0,8.0]])
covarianceMatrix(inputMatrix: [[1.0,2.0],[3.0,3.0]])
var m = [[1.0, 2.0], [3.0, 4.0]]
invert(inputMatrix: m)    // returns [-2.0, 1.0, 1.5, -0.5]
```

矩阵分解
1. 奇异值分解svd
2. 特征值分解ev

```swift
func svd(inputMatrix:Matrix) -> (u:Matrix, s:Matrix, v:Matrix) {
    let m = inputMatrix[0].count
    let n = inputMatrix.count
    let x = inputMatrix.reduce([], {$0+$1})
    var JOBZ = Int8(UnicodeScalar("A").value)
    var JOBU = Int8(UnicodeScalar("A").value)
    var JOBVT = Int8(UnicodeScalar("A").value)
    var M = __CLPK_integer(m)
    var N = __CLPK_integer(n)
    var A = x
    var LDA = __CLPK_integer(m)
    var S = [__CLPK_doublereal](repeating: 0.0, count: min(m,n))
    var U = [__CLPK_doublereal](repeating: 0.0, count: m*m)
    var LDU = __CLPK_integer(m)
    var VT = [__CLPK_doublereal](repeating: 0.0, count: n*n)
    var LDVT = __CLPK_integer(n)
    let lwork = min(m,n)*(6+4*min(m,n))+max(m,n)
    var WORK = [__CLPK_doublereal](repeating: 0.0, count: lwork)
    var LWORK = __CLPK_integer(lwork)
    var IWORK = [__CLPK_integer](repeating: 0, count: 8*min(m,n))
    var INFO = __CLPK_integer(0)
    if m >= n {
        dgesdd_(&JOBZ, &M, &N, &A, &LDA, &S, &U, &LDU, &VT, &LDVT, &WORK, &LWORK, &IWORK, &INFO)
    } else {
        dgesvd_(&JOBU, &JOBVT, &M, &N, &A, &LDA, &S, &U, &LDU, &VT, &LDVT, &WORK, &LWORK, &INFO)
    }
    var s = [Double](repeating: 0.0, count: m*n)
    for ni in 0...(min(m,n)-1) {
        s[ni*m+ni] = S[ni]
    }
    var v = [Double](repeating: 0.0, count: n*n)
    vDSP_mtransD(VT, 1, &v, 1, vDSP_Length(n), vDSP_Length(n))
    
    var outputU:Matrix = []
    var outputS:Matrix = []
    var outputV:Matrix = []
    for i in 0..<m {
        outputU.append(Array(U[i*m..<i*m+m]))
    }
    for i in 0..<n {
        outputS.append(Array(s[i*m..<i*m+m]))
    }
    for i in 0..<n {
        outputV.append(Array(v[i*n..<i*n+n]))
    }
    
    return (outputU, outputS, outputV)
}

func ev(inputMatrix: Matrix) -> (eigenvalues:Vector, eigenvectors:Matrix) {
    let n = inputMatrix.count
    let x = inputMatrix.reduce([], {$0+$1})
    var matrix:[__CLPK_doublereal] = x
    
    var N = __CLPK_integer(sqrt(Double(x.count)))
    var pivots = [__CLPK_integer](repeating: 0,count: Int(N))
    var workspaceQuery: Double = 0.0
    var error : __CLPK_integer = 0
    var lwork = __CLPK_integer(-1)
    var wr = [Double](repeating: 0.0, count: Int(N))
    var wi = [Double](repeating: 0.0, count: Int(N))
    var vl = [__CLPK_doublereal](repeating: 0.0, count: Int(N*N))
    var vr = [__CLPK_doublereal](repeating: 0.0, count: Int(N*N))
    
    dgeev_(UnsafeMutablePointer(mutating: ("V" as NSString).utf8String), UnsafeMutablePointer(mutating: ("V" as NSString).utf8String), &N, &matrix, &N, &wr, &wi, &vl, &N, &vr, &N, &workspaceQuery, &lwork, &error)
    var workspace = [Double](repeating: 0.0, count: Int(workspaceQuery))
    lwork = __CLPK_integer(workspaceQuery)
    
    dgeev_(UnsafeMutablePointer(mutating: ("V" as NSString).utf8String), UnsafeMutablePointer(mutating: ("V" as NSString).utf8String), &N, &matrix, &N, &wr, &wi, &vl, &N, &vr, &N, &workspace, &lwork, &error)
    
    var eigenvectors:Matrix = []
    for i in 0..<n {
        eigenvectors.append(Array(vl[i*n..<i*n+n]))
    }
    return (wr, eigenvectors)

}
```
