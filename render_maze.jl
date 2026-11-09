# Render an ASCII KDG maze to PNG.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
CLIPS_ROOT = joinpath(@__DIR__, "..", "CLIPS.jl")

using PDDL
using PDDLViz
using CairoMakie

PDDL.Arrays.@register()
include(joinpath(CLIPS_ROOT, "doors-keys-gems", "src", "ascii.jl"))

function get_obj_color(state::State, obj::Const)
    for color in PDDL.get_objects(state, :color)
        if state[Compound(:iscolor, Term[obj, color])]
            return color
        end
    end
    return Const(:none)
end

MAZE_PATH = joinpath(@__DIR__, ARGS[1])
DOMAIN_PATH = joinpath(CLIPS_ROOT, "doors-keys-gems", "dataset", "domain.pddl")
OUTPUT_PATH = joinpath(@__DIR__, ARGS[2])

"Read an ASCII maze, skipping blank lines and comments."
function load_maze_ascii(path::AbstractString)
    lines = String[]
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        push!(lines, s)
    end
    return join(lines, "\n")
end

println("Reading maze from: ", MAZE_PATH)
maze_ascii = load_maze_ascii(MAZE_PATH)
println("\nASCII maze:\n", maze_ascii, "\n")

problem = ascii_to_pddl(maze_ascii, "sample-maze")
domain = load_domain(DOMAIN_PATH)
state = initstate(domain, problem)

println("Loaded domain: ", PDDL.get_name(domain))
println("Objects: ", PDDL.get_objects(problem))
println("Goal: ", PDDL.get_goal(problem))
println("Human at: (", state[pddl"(xloc human)"], ", ", state[pddl"(yloc human)"], ")")
println("Robot at: (", state[pddl"(xloc robot)"], ", ", state[pddl"(yloc robot)"], ")")

using PDDLViz: to_color

vibrant = PDDLViz.colorschemes[:vibrant]
gem_colors = [to_color(:red), vibrant[2], colorant"#56b4e9", colorant"#009e73"]
colordict = Dict(
    :red => vibrant[1],
    :yellow => vibrant[2],
    :blue => colorant"#0072b2",
    :green => :springgreen,
    :purple => vibrant[5],
    :orange => vibrant[6],
    :pink => vibrant[5],
    :none => :gray,
)

renderer = PDDLViz.GridworldRenderer(
    resolution = (600, 700),
    has_agent = false,
    obj_renderers = Dict(
        :agent => (d, s, o) -> o.name == :human ?
            HumanGraphic() : RobotGraphic(),
        :key => (d, s, o) -> KeyGraphic(
            color = colordict[get_obj_color(s, o).name]
        ),
        :door => (d, s, o) -> LockedDoorGraphic(
            visible = s[Compound(:locked, [o])],
            color = colordict[get_obj_color(s, o).name]
        ),
        :gem => (d, s, o) -> GemGraphic(
            color = gem_colors[parse(Int, string(o.name)[end])]
        ),
    ),
    obj_type_z_order = [:door, :key, :gem, :agent],
    show_inventory = true,
    inventory_fns = [
        (d, s, o) -> s[Compound(:has, [Const(:human), o])],
        (d, s, o) -> s[Compound(:has, [Const(:robot), o])],
    ],
    inventory_types = [:item, :item],
    inventory_labels = ["Human Inventory", "Robot Inventory"],
)

println("\nRendering environment...")
canvas = renderer(domain, state)
save(OUTPUT_PATH, canvas)
println("Saved render to: ", OUTPUT_PATH)

# Show an interactive figure when a display is available
try
    display(canvas)
catch err
    @info "Could not open an interactive window ($(typeof(err))); PNG was still saved."
end

println("Done.")
