# Animate the independent-human value-iteration trajectory.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
CLIPS_ROOT = joinpath(@__DIR__, "..", "CLIPS.jl")

using PDDL
using PDDLViz
using CairoMakie
using JSON3

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
RESULTS_PATH = joinpath(@__DIR__, ARGS[2])
OUT_DIR = joinpath(@__DIR__, ARGS[3])

GOAL_COLORS = ["both"]
FRAMERATE = 2
TRAIL_LENGTH = 20


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

println("Reading maze: ", MAZE_PATH)
problem = ascii_to_pddl(load_maze_ascii(MAZE_PATH), "sample-maze")
domain = load_domain(DOMAIN_PATH)
state0 = initstate(domain, problem)

# Freeze the robot so the animation shows the human path only.
state0 = copy(state0)
state0[pddl"(frozen robot)"] = true


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
    resolution = (600, 720),
    has_agent = false,
    obj_renderers = Dict(
        :agent => (d, s, o) -> o.name == :human ?
            HumanGraphic() : RobotGraphic(visible = false),
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
    trajectory_options = Dict(
        :tracked_objects => [Const(:human)],
        :tracked_types => Const[],
        :object_colors => [:black],
    ),
)

isfile(RESULTS_PATH) || error("Missing $RESULTS_PATH")

results = JSON3.read(read(RESULTS_PATH, String))
println("Loaded results from: ", RESULTS_PATH)

"""Captions for human actions; robot-wait frames omit the action text."""
function build_captions(trajectory, pddl_plan, goal_color)
    captions = Dict{Int,String}()
    traj = collect(trajectory)
    start_u = traj[1]["utility"]
    captions[1] = "goal=$(goal_color)  t=0  V=$(start_u)  (start)"

    human_idx = 1
    for (step_i, act_str) in enumerate(pddl_plan)
        frame = step_i + 1
        act_s = String(act_str)

        if occursin("wait", act_s) || occursin("robot", act_s)
            if human_idx <= length(traj)
                step = traj[human_idx]
                captions[frame] = "goal=$(goal_color)  t=$(step["t"])  V=$(step["utility"])"
            else
                captions[frame] = "goal=$(goal_color)  t=0  V=$(start_u)"
            end
        else
            human_idx += 1
            if human_idx <= length(traj)
                step = traj[human_idx]
                act = step["action"]
                captions[frame] = "goal=$(goal_color)  t=$(step["t"])  V=$(step["utility"])  $(act)"
            end
        end
    end
    return captions
end
function animate_goal(goal_color::AbstractString)
    haskey(results["goals"], goal_color) ||
        error("Goal $goal_color not found in results JSON")

    sol = results["goals"][goal_color]
    traj = sol["trajectory"]
    pddl_plan_strs = sol["pddl_plan"]
    isempty(pddl_plan_strs) && error("Empty plan for goal $goal_color")

    plan = Term[PDDL.parse_pddl(String(a)) for a in pddl_plan_strs]
    captions = build_captions(traj, pddl_plan_strs, goal_color)

    println("\nAnimating goal=$(goal_color)  steps=$(length(plan))  path_cost=$(sol["path_cost"])")

    anim = anim_plan(
        renderer, domain, state0, plan;
        captions = captions,
        caption_color = :black,
        caption_size = 22,
        trail_length = TRAIL_LENGTH,
        framerate = FRAMERATE,
        format = "gif",
        loop = -1,
    )

    out_path = joinpath(OUT_DIR, "ind_H_trajectory.gif")
    save(out_path, anim)
    println("Saved: ", out_path)
    return out_path
end
mkpath(OUT_DIR)
for color in GOAL_COLORS
    animate_goal(color)
end

println("\nDone.")
