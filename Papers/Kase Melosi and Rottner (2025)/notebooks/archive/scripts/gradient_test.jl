using Test

@testset "_grad_sumsq / grad_l2" begin
    @test _grad_sumsq(nothing) == 0.0
    @test _grad_sumsq(3f0) == 9.0f0

    A = reshape(Float32.(1:6), 2, 3)
    b = Float32[1, -2, 3]

    @test _grad_sumsq(A) == sum(abs2, A)
    @test _grad_sumsq(b) == sum(abs2, b)

    g = (weight = A, bias = b)
    @test _grad_sumsq(g) == sum(abs2, A) + sum(abs2, b)

    g2 = (layer1 = g, layer2 = (w = Float32[0.5], b = nothing))
    @test _grad_sumsq(g2) == _grad_sumsq(g) + abs2(0.5f0)

    d = Dict(:w => A, :b => nothing)
    @test _grad_sumsq(d) == sum(abs2, A)

    @test grad_l2(g2) ≈ sqrt(_grad_sumsq(g2))

    # Original pitfall: applying abs2 to a container (should throw)
    @test_throws MethodError sum(abs2, g)
end