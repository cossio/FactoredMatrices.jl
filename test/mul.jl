using Test: @testset, @test, @test_throws
using LinearAlgebra: Adjoint, I, Transpose, dot, mul!, norm, pinv, rank, svd, tr
using FactoredMatrices: FactoredMatrix, FactoredMatrices

myrand(::Type{T}, dims::Integer...) where {T <: Real} = rand(T, dims...)
myrand(::Type{Complex{T}}, dims::Integer...) where {T <: Real} = rand(T, dims...) + im * rand(T, dims...)

# Wrappers so @allocated measures the call in a compiled context (and the keyword
# argument does not allocate a fresh NamedTuple at global scope).
alloc_mul(C, A, B, ws) = @allocated mul!(C, A, B; cache = ws)
alloc_mul5(C, A, B, α, β, ws) = @allocated mul!(C, A, B, α, β; cache = ws)
alloc_mul_nocache(C, A, B) = @allocated mul!(C, A, B)
alloc_mul_cached(C, A, B) = @allocated mul!(C, A, B)

for T in (Float32, Float64, ComplexF32, ComplexF64)
    # L: 15 × 10 with rank 3
    u = myrand(T, 15, 3)
    v = myrand(T, 10, 3)
    L = FactoredMatrix(u, v')
    M = Matrix(L)
    α, β = T(2), T(3)

    B = myrand(T, 10, 5) # for L * B
    D = myrand(T, 15, 5) # for L' * D
    E = myrand(T, 5, 15) # for E * L
    b = myrand(T, 10, 1)[:, 1]

    # 3-arg mul! into dense outputs
    C = zeros(T, 15, 5)
    @test mul!(C, L, B) ≈ M * B
    C = zeros(T, 10, 5)
    @test mul!(C, L', D) ≈ M' * D
    @test mul!(C, transpose(L), D) ≈ transpose(M) * D
    C = zeros(T, 5, 10)
    @test mul!(C, E, L) ≈ E * M
    c = zeros(T, 15)
    @test mul!(c, L, b) ≈ M * b
    d = myrand(T, 15, 1)[:, 1]
    c = zeros(T, 10)
    @test mul!(c, L', d) ≈ M' * d
    @test mul!(c, Transpose(L), conj(d)) ≈ transpose(M) * conj(d)

    # 5-arg mul! into dense outputs
    C0 = myrand(T, 15, 5)
    C = copy(C0)
    @test mul!(C, L, B, α, β) ≈ α * (M * B) + β * C0
    C0 = myrand(T, 10, 5)
    C = copy(C0)
    @test mul!(C, L', D, α, β) ≈ α * (M' * D) + β * C0
    C = copy(C0)
    @test mul!(C, Transpose(L), D, α, β) ≈ α * (transpose(M) * D) + β * C0
    C0 = myrand(T, 5, 10)
    C = copy(C0)
    @test mul!(C, E, L, α, β) ≈ α * (E * M) + β * C0
    C0 = myrand(T, 5, 15)
    C = copy(C0)
    G = myrand(T, 5, 10)
    @test mul!(C, G, Adjoint(L), α, β) ≈ α * (G * M') + β * C0
    c0 = myrand(T, 15, 1)[:, 1]
    c = copy(c0)
    @test mul!(c, L, b, α, β) ≈ α * (M * b) + β * c0

    # FM × FM into dense output, both association branches
    R = FactoredMatrix(myrand(T, 10, 2), myrand(T, 8, 2)') # rank(L) > rank(R)
    S = FactoredMatrix(myrand(T, 10, 4), myrand(T, 8, 4)') # rank(L) < rank(S)
    for X in (R, S)
        local C = zeros(T, 15, 8)
        @test mul!(C, L, X) ≈ M * Matrix(X)
        C0 = myrand(T, 15, 8)
        local Cm = copy(C0)
        @test mul!(Cm, L, X, α, β) ≈ α * (M * Matrix(X)) + β * C0
    end
    @test mul!(zeros(T, 8, 15), R', Adjoint(L)) ≈ Matrix(R)' * M'
    # wide low-rank right operand hits the other association branch
    W = FactoredMatrix(myrand(T, 10, 1), myrand(T, 20, 1)')
    @test mul!(zeros(T, 15, 20), L, W) ≈ M * Matrix(W)
    C0 = myrand(T, 15, 20)
    Cw = copy(C0)
    @test mul!(Cw, L, W, α, β) ≈ α * (M * Matrix(W)) + β * C0

    # mul! into a FactoredMatrix output
    Cf = FactoredMatrix(zeros(T, 15, 3), zeros(T, 5, 3)')
    @test Matrix(mul!(Cf, L, B)) ≈ M * B
    Cf = FactoredMatrix(zeros(T, 5, 3), zeros(T, 10, 3)')
    @test Matrix(mul!(Cf, E, L)) ≈ E * M
    Cf = FactoredMatrix(zeros(T, 10, 3), zeros(T, 5, 3)')
    @test Matrix(mul!(Cf, Adjoint(L), D)) ≈ M' * D
    # FM × FM → FM, landing on either factor depending on rank(C)
    Cf = FactoredMatrix(zeros(T, 15, 3), zeros(T, 8, 3)') # rank(C) == rank(L)
    @test Matrix(mul!(Cf, L, R)) ≈ M * Matrix(R)
    Cf = FactoredMatrix(zeros(T, 15, 2), zeros(T, 8, 2)') # rank(C) == rank(R)
    @test Matrix(mul!(Cf, L, R)) ≈ M * Matrix(R)
    Cf = FactoredMatrix(zeros(T, 15, 5), zeros(T, 8, 5)')
    @test_throws DimensionMismatch mul!(Cf, L, R)
    # dense × dense → FM repacks the factors
    A2 = myrand(T, 15, 3)
    B2 = myrand(T, 3, 8)
    Cf = FactoredMatrix(zeros(T, 15, 3), zeros(T, 8, 3)')
    @test Matrix(mul!(Cf, A2, B2)) ≈ A2 * B2
    # mismatched storage ranks must throw instead of broadcast-expanding the factors
    Cf2 = FactoredMatrix(zeros(T, 15, 2), zeros(T, 8, 2)')
    @test_throws DimensionMismatch mul!(Cf2, myrand(T, 15, 1), myrand(T, 1, 8))
    @test_throws DimensionMismatch mul!(Cf2, FactoredMatrix(myrand(T, 15, 1), myrand(T, 10, 1)'), myrand(T, 10, 8))
    # matching u factor but mismatched v factor
    Cf3 = FactoredMatrix(zeros(T, 15, 2), zeros(T, 9, 2)')
    @test_throws DimensionMismatch mul!(Cf3, myrand(T, 15, 2), myrand(T, 2, 8))

    # Workspace: correctness and zero allocations
    ws = FactoredMatrices.Workspace(L, 5)
    C = zeros(T, 15, 5)
    @test mul!(C, L, B; cache = ws) ≈ M * B
    C = zeros(T, 10, 5)
    @test mul!(C, L', D; cache = ws) ≈ M' * D
    C = zeros(T, 5, 10)
    @test mul!(C, E, L; cache = ws) ≈ E * M
    c = zeros(T, 15)
    @test mul!(c, L, b; cache = ws) ≈ M * b
    C0 = myrand(T, 15, 5)
    C = copy(C0)
    @test mul!(C, L, B, α, β; cache = ws) ≈ α * (M * B) + β * C0

    C = zeros(T, 15, 5)
    alloc_mul(C, L, B, ws)
    @test alloc_mul(C, L, B, ws) == 0
    # on Julia < 1.12, 5-arg mul! with runtime α, β constructs a boxed MulAddMul inside
    # LinearAlgebra (16-48 bytes depending on the eltype; plain dense mul! pays the
    # same); Julia 1.12 elides it
    alloc_mul5(C, L, B, α, β, ws)
    @test alloc_mul5(C, L, B, α, β, ws) ≤ (VERSION < v"1.12" ? 48 : 0)
    C = zeros(T, 10, 5)
    alloc_mul(C, L', D, ws)
    @test alloc_mul(C, L', D, ws) == 0
    C = zeros(T, 5, 10)
    alloc_mul(C, E, L, ws)
    @test alloc_mul(C, E, L, ws) == 0
    c = zeros(T, 15)
    alloc_mul(c, L, b, ws)
    @test alloc_mul(c, L, b, ws) == 0

    # FM × FM → FM with a rank(L) × rank(R) workspace is fully allocation-free
    wsr = FactoredMatrices.Workspace{T}(rank(L), rank(R))
    Cf = FactoredMatrix(zeros(T, 15, 3), zeros(T, 8, 3)')
    @test Matrix(mul!(Cf, L, R; cache = wsr)) ≈ M * Matrix(R)
    alloc_mul(Cf, L, R, wsr)
    @test alloc_mul(Cf, L, R, wsr) == 0
    # FM × dense → FM needs no cache to be allocation-free
    Cf = FactoredMatrix(zeros(T, 15, 3), zeros(T, 5, 3)')
    alloc_mul_nocache(Cf, L, B)
    @test alloc_mul_nocache(Cf, L, B) == 0

    # CachedFactoredMatrix
    cfm = FactoredMatrices.CachedFactoredMatrix(L, 5)
    @test size(cfm) == size(L)
    @test size(cfm, 1) == 15 && size(cfm, 2) == 10
    @test length(cfm) == length(L)
    @test rank(cfm) == rank(L)
    @test eltype(cfm) == T
    @test cfm[2, 3] == L[2, 3]
    @test Matrix(cfm) == Array(cfm) == M
    @test sum(abs2, cfm) ≈ sum(abs2, M)
    @test sprint(show, cfm) == "Cached" * sprint(show, L)
    @test sprint(show, MIME("text/plain"), cfm) == sprint(show, cfm)
    @test cfm * B ≈ M * B
    @test cfm * b ≈ M * b
    A5 = myrand(T, 5, 15)
    @test A5 * cfm ≈ A5 * M
    C = zeros(T, 15, 5)
    @test mul!(C, cfm, B) ≈ M * B
    C0 = myrand(T, 15, 5)
    C = copy(C0)
    @test mul!(C, cfm, B, α, β) ≈ α * (M * B) + β * C0
    C = zeros(T, 5, 10)
    @test mul!(C, A5, cfm) ≈ A5 * M
    C0 = myrand(T, 5, 10)
    C = copy(C0)
    @test mul!(C, A5, cfm, α, β) ≈ α * (A5 * M) + β * C0
    c = zeros(T, 15)
    @test mul!(c, cfm, b) ≈ M * b
    c0 = myrand(T, 15, 1)[:, 1]
    c = copy(c0)
    @test mul!(c, cfm, b, α, β) ≈ α * (M * b) + β * c0
    C = zeros(T, 15, 5)
    alloc_mul_cached(C, cfm, B)
    @test alloc_mul_cached(C, cfm, B) == 0
end

# cached products with a promoting operand fall back to allocating intermediates
# instead of writing promoted values into the typed buffers
L = FactoredMatrix(randn(6, 2), randn(5, 2)')
cfm = FactoredMatrices.CachedFactoredMatrix(L, 4)
Bc = randn(ComplexF64, 5, 4)
xc = randn(ComplexF64, 5)
Ac = randn(ComplexF64, 4, 6)
@test cfm * Bc ≈ Matrix(L) * Bc
@test eltype(cfm * Bc) == ComplexF64
@test cfm * xc ≈ Matrix(L) * xc
@test Ac * cfm ≈ Ac * Matrix(L)
C = zeros(ComplexF64, 6, 4)
@test mul!(C, cfm, Bc) ≈ Matrix(L) * Bc
@test mul!(C, cfm, Bc, 2.0, 1.0) ≈ 3 * Matrix(L) * Bc
c = zeros(ComplexF64, 6)
@test mul!(c, cfm, xc) ≈ Matrix(L) * xc
C = zeros(ComplexF64, 4, 5)
@test mul!(C, Ac, cfm) ≈ Ac * Matrix(L)
@test mul!(C, Ac, cfm, 2.0, 1.0) ≈ 3 * Ac * Matrix(L)
c0 = randn(ComplexF64, 6)
c = copy(c0)
@test mul!(c, cfm, xc, 2.0, 3.0) ≈ 2 * (Matrix(L) * xc) + 3 * c0

# factored operands: cached products delegate to the lazy factored products
R = FactoredMatrix(randn(5, 3), randn(7, 3)') # L is 6 × 5, so L * R is 6 × 7
G = FactoredMatrix(randn(4, 2), randn(6, 2)') # G * L is 4 × 5
W = FactoredMatrix(randn(7, 2), randn(5, 2)') # L * W' is 6 × 7
@test cfm * R isa FactoredMatrix
@test Matrix(cfm * R) ≈ Matrix(L) * Matrix(R)
@test G * cfm isa FactoredMatrix
@test Matrix(G * cfm) ≈ Matrix(G) * Matrix(L)
@test Matrix(cfm * Adjoint(W)) ≈ Matrix(L) * Matrix(W)'
@test Matrix(Adjoint(G') * cfm) ≈ Matrix(G) * Matrix(L)
@test Matrix(cfm * FactoredMatrices.CachedFactoredMatrix(R, 2)) ≈ Matrix(L) * Matrix(R)

# cached mul! with factored operands; the bundled buffer is used only when it fits
cfmR = FactoredMatrices.CachedFactoredMatrix(L, rank(R)) # left buffer fits L × R products
for A in (cfmR, cfm) # fitting and non-fitting buffers give the same result
    local C = zeros(6, 7)
    @test mul!(C, A, Adjoint(W)) ≈ Matrix(L) * Matrix(W)'
    @test mul!(C, A, R) ≈ Matrix(L) * Matrix(R)
    @test mul!(C, A, R, 2.0, 1.0) ≈ 3 * Matrix(L) * Matrix(R) # uses C = L * R from above
    local Cf = FactoredMatrix(zeros(6, 3), zeros(7, 3)')
    @test Matrix(mul!(Cf, A, R)) ≈ Matrix(L) * Matrix(R)
end
C = zeros(4, 5)
@test mul!(C, G, cfm) ≈ Matrix(G) * Matrix(L)
@test mul!(C, Adjoint(G'), cfm) ≈ Matrix(G) * Matrix(L)
@test mul!(C, G, cfm, 2.0, 1.0) ≈ 3 * Matrix(G) * Matrix(L)
Cf = FactoredMatrix(zeros(4, 2), zeros(5, 2)')
@test Matrix(mul!(Cf, G, cfm)) ≈ Matrix(G) * Matrix(L)
C = zeros(6, 7)
@test mul!(C, cfmR, FactoredMatrices.CachedFactoredMatrix(R, 2)) ≈ Matrix(L) * Matrix(R)
@test mul!(C, cfmR, FactoredMatrices.CachedFactoredMatrix(R, 2), 2.0, 1.0) ≈ 3 * Matrix(L) * Matrix(R)
Cf = FactoredMatrix(zeros(6, 3), zeros(7, 3)')
@test Matrix(mul!(Cf, cfmR, FactoredMatrices.CachedFactoredMatrix(R, 2))) ≈ Matrix(L) * Matrix(R)

# an explicit Workspace whose element type cannot hold the intermediate falls back to
# allocating instead of throwing InexactError
wse = FactoredMatrices.Workspace(L, 4)
Bc = randn(ComplexF64, 5, 4)
xc = randn(ComplexF64, 5)
Ac = randn(ComplexF64, 4, 6)
@test mul!(zeros(ComplexF64, 6, 4), L, Bc; cache = wse) ≈ Matrix(L) * Bc
@test mul!(zeros(ComplexF64, 6), L, xc; cache = wse) ≈ Matrix(L) * xc
@test mul!(zeros(ComplexF64, 4, 5), Ac, L; cache = wse) ≈ Ac * Matrix(L)

# factored × cached with equal ranks uses the left buffer directly
cfmEq = FactoredMatrices.CachedFactoredMatrix(L, 2) # left buffer is 2 × 2 = rank(G) × rank(L)
@test mul!(zeros(4, 5), G, cfmEq) ≈ Matrix(G) * Matrix(L)

# factored × cached with unequal ranks reuses the (transposed-shape) right buffer
A3 = FactoredMatrix(randn(4, 3), randn(6, 3)')
cfm3 = FactoredMatrices.CachedFactoredMatrix(L, 3) # right buffer is 3 × 2 = rank(A3) × rank(L)
C45 = zeros(4, 5)
@test mul!(C45, A3, cfm3) ≈ Matrix(A3) * Matrix(L)
Cf45 = FactoredMatrix(zeros(4, 3), zeros(5, 3)')
@test Matrix(mul!(Cf45, A3, cfm3)) ≈ Matrix(A3) * Matrix(L)

# in-place products over an empty contracted dimension are exactly zero too
Le = FactoredMatrix(fill(Inf, 2, 1), zeros(0, 1)') # 2 × 0
Me = FactoredMatrix(zeros(0, 1), fill(Inf, 3, 1)') # 0 × 3
@test iszero(mul!(randn(2, 3), Le, Me))
C0 = randn(2, 3)
@test mul!(copy(C0), Le, Me, 2.0, 3.0) ≈ 3 * C0
@test iszero(mul!(randn(2, 3), Le, zeros(0, 3)))
@test iszero(mul!(randn(3, 3), zeros(3, 0), Me))
@test iszero(mul!(randn(2), Le, zeros(0)))
Cfe = FactoredMatrix(randn(2, 1), randn(3, 1)')
@test iszero(Matrix(mul!(Cfe, Le, Me)))
@test iszero(Matrix(mul!(FactoredMatrix(randn(2, 1), randn(3, 1)'), Le, zeros(0, 3))))
@test iszero(Matrix(mul!(FactoredMatrix(randn(3, 1), randn(3, 1)'), zeros(3, 0), Me)))
@test_throws DimensionMismatch mul!(FactoredMatrix(randn(5, 1), randn(3, 1)'), Le, Me)

# adjoint/transpose of a cached matrix forward to the wrapped matrix, sharing buffers
@test cfm' isa FactoredMatrices.CachedFactoredMatrix
@test transpose(cfm) isa FactoredMatrices.CachedFactoredMatrix
x6 = randn(6)
@test cfm' * x6 ≈ Matrix(L)' * x6
@test Matrix(transpose(cfm)) ≈ transpose(Matrix(L))
y5 = zeros(5)
@test mul!(y5, cfm', x6) ≈ Matrix(L)' * x6

# scalar products forward, keeping the buffers when the element type is preserved
@test 2 * cfm isa FactoredMatrices.CachedFactoredMatrix
@test Matrix(2 * cfm) ≈ 2 * Matrix(L)
@test Matrix(cfm * 3) ≈ 3 * Matrix(L)
@test Matrix(cfm / 2) ≈ Matrix(L) / 2
@test Matrix(2 \ cfm) ≈ Matrix(L) / 2
@test (1 + im) * cfm isa FactoredMatrices.CachedFactoredMatrix # buffers just go unused
@test Matrix((1 + im) * cfm) ≈ (1 + im) * Matrix(L)
@test cfm * I isa FactoredMatrices.CachedFactoredMatrix
@test Matrix(cfm * I) ≈ Matrix(L)
@test Matrix(I * cfm) ≈ Matrix(L)
@test Matrix(cfm * (2I)) ≈ 2 * Matrix(L)

# norm and least-squares solves forward to the wrapped factorization
@test norm(cfm) ≈ norm(Matrix(L))
@test cfm \ x6 ≈ pinv(Matrix(L)) * x6
@test cfm \ ((1 + im) * x6) ≈ pinv(Matrix(L)) * ((1 + im) * x6)

# svd, dot and tr forward to the wrapped factorization
@test svd(cfm).S == svd(L).S
M65 = Matrix(L)
@test dot(cfm, cfm) ≈ dot(M65, M65)
@test dot(cfm, L) ≈ dot(M65, M65)
@test dot(L, cfm) ≈ dot(M65, M65)
@test dot(cfm, M65) ≈ dot(M65, M65)
@test dot(M65, cfm) ≈ dot(M65, M65)
Lsq = FactoredMatrix(randn(5, 2), randn(5, 2)')
@test tr(FactoredMatrices.CachedFactoredMatrix(Lsq, 2)) ≈ tr(Matrix(Lsq))

# row-vector products keep their row-vector shape through the cache
@test x6' * cfm isa Adjoint{<:Any, <:AbstractVector}
@test x6' * cfm ≈ x6' * M65
@test transpose(x6) * cfm isa Transpose{<:Any, <:AbstractVector}
@test transpose(x6) * cfm ≈ transpose(x6) * M65

# iszero forwards to the wrapped factorization
@test !iszero(cfm)
@test iszero(FactoredMatrices.CachedFactoredMatrix(FactoredMatrix(zeros(3, 1), zeros(2, 1)'), 1))

# comparisons forward to the wrapped factorization, ignoring the workspace
cfmB = FactoredMatrices.CachedFactoredMatrix(copy(L), 3)
@test cfm == cfmB
@test isequal(cfm, cfmB)
@test hash(cfm) == hash(cfmB)
@test cfm ≈ cfmB
@test cfm == L
@test L == cfm
@test isequal(cfm, L)
@test isequal(L, cfm)
@test cfm ≈ L
@test L ≈ cfm

# lazy sums and differences forward to the wrapped factorization
@test cfm + cfm isa FactoredMatrix
@test Matrix(cfm + cfm) ≈ 2 * M65
@test Matrix(cfm + L) ≈ 2 * M65
@test Matrix(L + cfm) ≈ 2 * M65
@test Matrix(cfm - L) ≈ zeros(6, 5) atol = 1.0e-8
@test Matrix(L - cfm) ≈ zeros(6, 5) atol = 1.0e-8
@test Matrix(cfm - cfm) ≈ zeros(6, 5) atol = 1.0e-8
@test +cfm === cfm
@test -cfm isa FactoredMatrices.CachedFactoredMatrix
@test Matrix(-cfm) ≈ -M65

# copies duplicate the factors and get independent scratch buffers
cfmCopy = copy(cfm)
@test cfmCopy isa FactoredMatrices.CachedFactoredMatrix
@test cfmCopy == cfm
@test cfmCopy.M.u !== cfm.M.u
@test cfmCopy.ws !== cfm.ws
@test cfmCopy.ws.left !== cfm.ws.left
@test size(cfmCopy.ws.left) == size(cfm.ws.left)

# factored destinations work with dense operands through the cache
mfB = randn(5, 4)
CfL = FactoredMatrix(zeros(6, 2), zeros(4, 2)')
@test Matrix(mul!(CfL, cfm, mfB)) ≈ M65 * mfB
mfA = randn(4, 6)
CfR = FactoredMatrix(zeros(4, 2), zeros(5, 2)')
@test Matrix(mul!(CfR, mfA, cfm)) ≈ mfA * M65

# a workspace with a wider element type than the matrix can be bundled; it serves the
# products whose intermediates are complex, while real products fall back to allocating
# (a complex buffer would silently widen the arithmetic of a real product)
wsc = FactoredMatrices.Workspace{ComplexF64}(2, 4)
cfmc = FactoredMatrices.CachedFactoredMatrix(L, wsc)
B4 = randn(5, 4)
@test cfmc * B4 ≈ Matrix(L) * B4
@test eltype(cfmc * B4) == Float64
@test mul!(zeros(6, 4), L, B4; cache = wsc) ≈ Matrix(L) * B4
B4c = randn(ComplexF64, 5, 4)
@test cfmc * B4c ≈ Matrix(L) * B4c
@test eltype(cfmc * B4c) == ComplexF64

# Boolean products accumulate into Int, like ordinary matrix products; the convenience
# Workspace constructor therefore allocates Int buffers, so they are actually usable
Lb = FactoredMatrix(trues(1, 2), trues(1, 2)')
@test FactoredMatrices.Workspace(Lb, 1) isa FactoredMatrices.Workspace{Int}
cfb = FactoredMatrices.CachedFactoredMatrix(Lb, 1)
@test cfb * trues(1, 1) == fill(2, 1, 1)
@test eltype(cfb * trues(1, 1)) == Int
@test cfb * trues(1) == fill(2, 1)
@test trues(1, 1) * cfb == fill(2, 1, 1)

# Workspace constructors
ws = FactoredMatrices.Workspace{Float64}(3, 7)
@test size(ws.left) == (3, 7)
@test size(ws.right) == (7, 3)
L = FactoredMatrix(randn(6, 2), randn(5, 2)')
ws = FactoredMatrices.Workspace(L, 4)
@test size(ws.left) == (2, 4)
@test size(ws.right) == (4, 2)
