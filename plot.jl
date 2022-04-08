using JLD2
using Statistics
using StatsPlots

path = "data/kernel_5_3_100_0.1.jld2"
f = jldopen(path, "r")

eigmin_H = f["eigmin_H"]
eigmin_H1 = f["eigmin_H1"]
eigmin_H2 = f["eigmin_H2"]
eigmin_H3 = f["eigmin_H3"]

eigmin_H1_H2 = f["eigmin_H1_H2"]
eigmin_H1_H3 = f["eigmin_H1_H3"]
eigmin_H2_H3 = f["eigmin_H2_H3"]

println("Num runs = $(size(eigmin_H)[1])")
println("Min eigmin_H = $(minimum(eigmin_H))")
println("Avg eigmin_H = $(mean(eigmin_H))")
println("Min eigmin_H1 = $(minimum(eigmin_H1))")
println("Min eigmin_H2 = $(minimum(eigmin_H2))")
println("Min eigmin_H3 = $(minimum(eigmin_H3))")

println("Min eigmin_H1_H2 = $(minimum(eigmin_H1_H2))")
println("Min eigmin_H1_H3 = $(minimum(eigmin_H1_H3))")
println("Min eigmin_H2_H3 = $(minimum(eigmin_H2_H3))")
