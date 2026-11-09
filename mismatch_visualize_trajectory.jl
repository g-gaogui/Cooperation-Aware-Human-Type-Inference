# Animate the mixed-policy (independent human, collaborative assistant) trajectory.

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

GOAL_LABELS = nothing  # nothing = all goals in the JSON
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

isfile(RESULTS_PATH) || error("Missing $RESULTS_PATH")

results = JSON3.read(read(RESULTS_PATH, String))
println("Loaded mismatch results from: ", RESULTS_PATH)

"""Map trajectory[i] onto anim_plan frame i (frame 1 is the initial state)."""
function build_joint_captions(trajectory, goal_label)
    captions = Dict{Int,String}()
    traj = collect(trajectory)

    step0 = traj[1]
    captions[1] = "goal=$(goal_label)  t=0  V=$(round(step0["utility"]))  " *
        "HC=$(round(step0["human_cost_to_go"]))  AC=$(round(step0["assistant_cost_to_go"]))  (start)"

    for i in 2:length(traj)
        step = traj[i]
        act = step["action"]
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

    out_path = joinpath(OUT_DIR, "mismatch_trajectory.gif")
    save(out_path, anim)
    println("Saved: ", out_path)
    return out_path
end

mkpath(OUT_DIR)
labels = GOAL_LABELS === nothing ? collect(keys(results["goals"])) : GOAL_LABELS
for label in labels
    animate_goal(String(label))
end

println("\nDone.")