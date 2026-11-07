#=
Step 1: Load a maze from ASCII and render it.

ASCII legend (Doors, Keys & Gems):
  . = empty cell
  W = wall
  h = human agent
  m = robot agent
  r/y/b/e/p = red / yellow / blue / green / pink key
  R/Y/B/E/P = red / yellow / blue / green / pink locked door
  g = gem

Run from the repo root (or this folder) in the Julia REPL:
  include("my_project/render_maze.jl")
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using PDDL
using PDDLViz
using CairoMakie  # headless-friendly; use GLMakie for an interactive window

# Register PDDL array extension (needed for the walls bit-matrix)
PDDL.Arrays.@register()

# Reuse the repo's ASCII → PDDL converter
include(joinpath(@__DIR__, "..", "doors-keys-gems", "src", "ascii.jl"))

# Minimal helpers used by the renderer (from doors-keys-gems/src/utils.jl)
function get_obj_color(state::State, obj::Const)
    for color in PDDL.get_objects(state, :color)
        if state[Compound(:iscolor, Term[obj, color])]
            return color
        end
    end
    return Const(:none)
end

## Paths ##


MAZE_PATH = joinpath(@__DIR__, ARGS[1])
DOMAIN_PATH = joinpath(@__DIR__, "..", "doors-keys-gems", "dataset", "domain.pddl")
OUTPUT_PATH = joinpath(@__DIR__, ARGS[2])

## Load maze ##

"Read an ASCII maze file, ignoring blank lines and lines starting with '#'."
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

# Convert ASCII layout → PDDL problem, then initialize a state
problem = ascii_to_pddl(maze_ascii, "sample-maze")
domain = load_domain(DOMAIN_PATH)
state = initstate(domain, problem)

println("Loaded domain: ", PDDL.get_name(domain))
println("Objects: ", PDDL.get_objects(problem))
println("Goal: ", PDDL.get_goal(problem))
println("Human at: (", state[pddl"(xloc human)"], ", ", state[pddl"(yloc human)"], ")")
println("Robot at: (", state[pddl"(xloc robot)"], ", ", state[pddl"(yloc robot)"], ")")

## Renderer (same style as doors-keys-gems/src/render.jl) ##

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

## Render ##

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
