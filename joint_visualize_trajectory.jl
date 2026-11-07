#=
Animate the JOINT human+assistant value-iteration trajectory with
utility / per-agent-cost captions.

Differs from visualize_trajectory.jl (step 1, human-only) in three
ways:
  1. The robot is NOT frozen — it takes its own real actions, so both
     agents move.
  2. Captions show combined utility AND the human/assistant cost-to-go
     split (human_cost_to_go / assistant_cost_to_go from the joint
     results JSON), plus which agent acted ("H:..." / "A:...").
  3. `results["goals"]` keys are joint goal labels like
     "H:red+A:yellow" rather than a single gem color, and trajectory
     entries align 1:1 with pddl_plan steps (no interleaved
     "(wait robot)" padding to skip over), since joint_value_iteration.py
     already writes exactly one trajectory entry per real pddl_plan
     action.

Prereq:
  python my_project/joint_value_iteration.py
      # writes mazes/joint_value_iteration_results.json

Run from repo root in Julia (project activated, Julia 1.10/1.11):
  include("my_project/visualize_joint_trajectory.jl")

Outputs (one per goal label found in the results JSON, unless
GOAL_LABELS is set below to a subset):
  my_project/mazes/joint_traj_<sanitized-goal-label>.gif
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using PDDL
using PDDLViz
using CairoMakie
using JSON3

PDDL.Arrays.@register()

include(joinpath(@__DIR__, "..", "doors-keys-gems", "src", "ascii.jl"))

function get_obj_color(state::State, obj::Const)
    for color in PDDL.get_objects(state, :color)
        if state[Compound(:iscolor, Term[obj, color])]
            return color
        end
    end
    return Const(:none)
end

## Paths / options ##

MAZE_PATH = joinpath(@__DIR__, ARGS[1])
DOMAIN_PATH = joinpath(@__DIR__, "..", "doors-keys-gems", "dataset", "domain.pddl")
RESULTS_PATH = joinpath(@__DIR__, ARGS[2])
OUT_DIR = joinpath(@__DIR__, ARGS[3])

# Which goal labels to animate. `nothing` means "animate every goal
# found in results["goals"]"; set to e.g. ["H:red+A:yellow"] to pick
# specific ones.
GOAL_LABELS = nothing
FRAMERATE = 2
TRAIL_LENGTH = 20

## Load maze → PDDL ##

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

# NOTE: unlike the step-1 visualizer, we do NOT freeze the robot here —
# it has real planned actions and should move.

## Renderer ##

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
    trajectory_options = Dict(
        :tracked_objects => [Const(:human), Const(:robot)],
        :tracked_types => Const[],
        :object_colors => [:black, :gray],
    ),
)

## Load joint VI results ##

isfile(RESULTS_PATH) || error(
    "Missing $RESULTS_PATH\nRun: python my_project/joint_value_iteration.py"
)

results = JSON3.read(read(RESULTS_PATH, String))
println("Loaded joint results from: ", RESULTS_PATH)

"""
Build caption dict for anim_plan frames (1-indexed). Trajectory
entries and pddl_plan steps are already 1:1 aligned by
joint_value_iteration.py — trajectory[1] is the start (before any
action), trajectory[i+1] is the state after pddl_plan[i] — so frame
i (1-indexed, matching PDDLViz's anim_plan convention where frame 1
is the initial state) maps directly onto trajectory[i].
"""
function build_joint_captions(trajectory, goal_label)
    captions = Dict{Int,String}()
    traj = collect(trajectory)

    step0 = traj[1]
    captions[1] = "goal=$(goal_label)  t=0  V=$(round(step0["utility"]))  " *
        "HC=$(round(step0["human_cost_to_go"]))  AC=$(round(step0["assistant_cost_to_go"]))  (start)"

    for i in 2:length(traj)
        step = traj[i]
        act = step["action"]  # e.g. "H:right" or "A:pickup(gem:yellow)"
        captions[i] = "goal=$(goal_label)  t=$(step["t"])  V=$(round(step["utility"]))  " *
            "HC=$(round(step["human_cost_to_go"]))  AC=$(round(step["assistant_cost_to_go"]))  $(act)"
    end
    return captions
end

sanitize(label::AbstractString) = replace(String(label), r"[^A-Za-z0-9]+" => "_")

function animate_goal(goal_label::AbstractString)
    haskey(results["goals"], goal_label) ||
        error("Goal $goal_label not found in results JSON")

    sol = results["goals"][goal_label]
    traj = sol["trajectory"]
    pddl_plan_strs = sol["pddl_plan"]
    isempty(pddl_plan_strs) && error("Empty plan for goal $goal_label")
    length(traj) == length(pddl_plan_strs) + 1 || @warn(
        "trajectory/pddl_plan length mismatch for $goal_label " *
        "(expected trajectory to have exactly one more entry than pddl_plan)"
    )

    plan = Term[PDDL.parse_pddl(String(a)) for a in pddl_plan_strs]
    captions = build_joint_captions(traj, goal_label)

    println(
        "\nAnimating goal=$(goal_label)  steps=$(length(plan))  " *
        "combined_cost=$(sol["path_cost"])  human_cost=$(sol["human_path_cost"])  " *
        "assistant_cost=$(sol["assistant_path_cost"])"
    )
    for step in traj
        println(
            "  t=$(step["t"])  turn=$(step["turn"])  H=($(step["x"]),$(step["y"]))  " *
            "A=($(step["rx"]),$(step["ry"]))  V=$(step["utility"])  $(step["action"])"
        )
    end

    anim = anim_plan(
        renderer, domain, state0, plan;
        captions = captions,
        caption_color = :black,
        caption_size = 20,
        trail_length = TRAIL_LENGTH,
        framerate = FRAMERATE,
        format = "gif",
        loop = -1,
    )

    out_path = joinpath(OUT_DIR, "col_joint_trajectory.gif")
    save(out_path, anim)
    println("Saved: ", out_path)
    return out_path
end

## Run ##

mkpath(OUT_DIR)
labels = GOAL_LABELS === nothing ? collect(keys(results["goals"])) : GOAL_LABELS
for label in labels
    animate_goal(String(label))
end

println("\nDone.")