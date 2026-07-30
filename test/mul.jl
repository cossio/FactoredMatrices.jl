using Test: @testset, @test, @test_throws
using LinearAlgebra: Adjoint, Transpose, mul!, rank
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

# Boolean products accumulate into Int, like ordinary matrix products
Lb = FactoredMatrix(trues(1, 2), trues(1, 2)')
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
