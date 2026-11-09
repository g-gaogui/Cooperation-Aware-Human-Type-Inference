# Animate a predicted trajectory with per-step type-inference captions.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
CLIPS_ROOT = joinpath(@__DIR__, "..", "CLIPS.jl")

using PDDL
using PDDLViz
using CairoMakie
using JSON3
using Printf

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

DOMAIN_PATH = joinpath(
    CLIPS_ROOT,
    "doors-keys-gems",
    "dataset",
    "domain.pddl",
)

PREDICTION_PATH = joinpath(
    @__DIR__,
    ARGS[2],
)

OUT_DIR = joinpath(@__DIR__,  ARGS[3])
GIF_PATH = joinpath(OUT_DIR, ARGS[4])
PNG_PATH = joinpath(OUT_DIR, ARGS[5])

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

problem = ascii_to_pddl(
    load_maze_ascii(MAZE_PATH),
    "sample-maze",
)

domain = load_domain(DOMAIN_PATH)

state0 = initstate(domain, problem)

state0 = copy(state0)
state0[pddl"(frozen robot)"] = true


using PDDLViz: to_color

vibrant = PDDLViz.colorschemes[:vibrant]

gem_colors = [
    to_color(:red),
    vibrant[2],
    colorant"#56b4e9",
    colorant"#009e73",
]

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
        :agent => (d, s, o) ->
            o.name == :human ?
                HumanGraphic() :
                RobotGraphic(visible = true),

        :key => (d, s, o) ->
            KeyGraphic(
                color = colordict[get_obj_color(s, o).name]
            ),

        :door => (d, s, o) ->
            LockedDoorGraphic(
                visible = s[Compound(:locked, [o])],
                color = colordict[get_obj_color(s, o).name]
            ),

        :gem => (d, s, o) ->
            GemGraphic(
                color = gem_colors[
                    parse(Int, string(o.name)[end])
                ]
            ),
    ),

    obj_type_z_order = [
        :door,
        :key,
        :gem,
        :agent,
    ],

    show_inventory = true,

    inventory_fns = [
        (d, s, o) ->
            s[Compound(:has, [Const(:human), o])],

        (d, s, o) ->
            s[Compound(:has, [Const(:robot), o])],
    ],

    inventory_types = [
        :item,
        :item,
    ],

    inventory_labels = [
        "Human Inventory",
        "Robot Inventory",
    ],

    trajectory_options = Dict(
        :tracked_objects => [Const(:human)],
        :tracked_types => Const[],
        :object_colors => [:black],
    ),
)


isfile(PREDICTION_PATH) || error(
    """
    Missing prediction file:

        $PREDICTION_PATH

    Run the trajectory prediction / inference code first.
    """
)

prediction = JSON3.read(
    read(PREDICTION_PATH, String)
)

println("Loaded prediction from: ", PREDICTION_PATH)


function format_probability(x)
    if x === nothing
        return "N/A"
    end

    return @sprintf("%.1f%%", 100 * Float64(x))
end


function json_string(x)
    return String(x)
end


function build_plan(trajectory)

    plan = Term[]

    traj = collect(trajectory)

    for step in traj

        if !haskey(step, "action") || step["action"] === nothing
            continue
        end

        action = String(step["action"])

        if haskey(step, "pddl_action") &&
           step["pddl_action"] !== nothing

            pddl_action = String(step["pddl_action"])

        else
            error(
                """
                Prediction trajectory entry for action
                '$action' does not contain 'pddl_action'.

                Please make sure the prediction generator stores
                the PDDL action for each human action.
                """
            )
        end

        push!(
            plan,
            PDDL.parse_pddl(pddl_action),
        )

        push!(
            plan,
            PDDL.parse_pddl("(wait robot)"),
        )
    end

    return plan
end


function build_captions(
    trajectory,
    goal_color,
)

    captions = Dict{Int,String}()

    traj = collect(trajectory)

    isempty(traj) &&
        error("Prediction trajectory is empty")

    first_step = traj[1]

    t0 = haskey(first_step, "t") ?
        Int(first_step["t"]) :
        0

    p_i0 =
        haskey(first_step, "p_independent") ?
            format_probability(first_step["p_independent"]) :
            "N/A"

    p_c0 =
        haskey(first_step, "p_collaborative") ?
            format_probability(first_step["p_collaborative"]) :
            "N/A"

    posterior0 =
        haskey(first_step, "posterior_collaborative") ?
            format_probability(first_step["posterior_collaborative"]) :
            "N/A"

    captions[1] =
        "P(a|I)=$(p_i0)  " *
        "P(a|C)=$(p_c0)  " *
        "P(C|τ)=$(posterior0)  (start)"


    human_idx = 1

    for step in traj

        if !haskey(step, "action") ||
           step["action"] === nothing

            continue
        end

        human_idx += 1

        if human_idx > length(traj)
            break
        end

        current_step = traj[human_idx]

        frame_human = 2 * (human_idx - 1)

        t =
            haskey(current_step, "t") ?
                Int(current_step["t"]) :
                human_idx - 1

        p_i =
            haskey(current_step, "p_independent") ?
                format_probability(
                    current_step["p_independent"]
                ) :
                "N/A"

        p_c =
            haskey(current_step, "p_collaborative") ?
                format_probability(
                    current_step["p_collaborative"]
                ) :
                "N/A"

        posterior =
            haskey(current_step, "posterior_collaborative") ?
                format_probability(
                    current_step["posterior_collaborative"]
                ) :
                "N/A"

        action =
            haskey(current_step, "action") &&
            current_step["action"] !== nothing ?
                String(current_step["action"]) :
                ""


        # ----------------------------------------------------
        # Human-action frame
        # ----------------------------------------------------

        captions[frame_human] =
            # "goal=$(goal_color)  " *
            "t=$(t)  " *
            "P(a|I)=$(p_i)  " *
            "P(a|C)=$(p_c)  " *
            "P(C|τ)=$(posterior)  " *
            "$(action)"


        # ----------------------------------------------------
        # Robot-wait frame
        # ----------------------------------------------------

        frame_robot = frame_human + 1

        captions[frame_robot] =
            # "goal=$(goal_color)  " *
            "t=$(t)  " *
            "P(a|I)=$(p_i)  " *
            "P(a|C)=$(p_c)  " *
            "P(C|τ)=$(posterior)"
    end

    return captions
end


## ------------------------------------------------------------
## Animate prediction
## ------------------------------------------------------------

function animate_prediction()

    if !haskey(prediction, "trajectory")
        error(
            """
            Prediction JSON does not contain a 'trajectory' field.

            Expected format:

            {
                "trajectory": [
                    ...
                ]
            }
            """
        )
    end

    trajectory = prediction["trajectory"]

    isempty(trajectory) &&
        error("Prediction trajectory is empty")

    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    goal_color =
        haskey(prediction, "goal") ?
            String(prediction["goal"]) :
            "unknown"

    # --------------------------------------------------------
    # Build plan
    # --------------------------------------------------------

    plan = build_plan(trajectory)

    isempty(plan) &&
        error("Empty predicted plan")

    # --------------------------------------------------------
    # Build captions
    # --------------------------------------------------------

    captions = build_captions(
        trajectory,
        goal_color,
    )

    println(
        "\nAnimating predicted trajectory"
    )

    println(
        "  goal = ",
        goal_color,
    )

    println(
        "  human actions = ",
        length(plan) ÷ 2,
    )

    println(
        "  animation frames = ",
        length(plan),
    )

    # --------------------------------------------------------
    # Animate
    # --------------------------------------------------------

    anim = anim_plan(
        renderer,
        domain,
        state0,
        plan;

        captions = captions,

        caption_color = :black,

        caption_size = 20,

        trail_length = TRAIL_LENGTH,

        framerate = FRAMERATE,

        format = "gif",

        loop = -1,
    )

    out_path = joinpath(
        GIF_PATH,
    )

    save(
        out_path,
        anim,
    )

    println(
        "Saved: ",
        out_path,
    )

    return out_path
end





function plot_probability_over_time()
    trajectory = prediction["trajectory"]

    # Extract per-timestep collaborative posterior from the trajectory,
    # not the scalar final posterior at the top level.
    posterior = Float64[step["posterior_collaborative"] for step in trajectory]

    # Independent probability is the complement of collaborative probability.
    p_independent = 1.0 .- posterior
    p_collaborative = posterior

    # Time points: one per trajectory entry (t=0 is the initial state).
    t = Int[step["t"] for step in trajectory]

    fig = Figure(size=(800, 500))

    ax = Axis(
        fig[1, 1],
        xlabel="Time step",
        ylabel="Probability",
        title="Human Type Inference Over Time",
        limits=(nothing, (0, 1)),
    )

    lines!(
        ax,
        t,
        p_independent,
        label="Independent",
        linewidth=3,
    )

    lines!(
        ax,
        t,
        p_collaborative,
        label="Collaborative",
        linewidth=3,
    )

    axislegend(ax, position=:rb)

    output_path = joinpath(PNG_PATH)
    save(output_path, fig)

    println("Saved probability plot to: ", output_path)

    return output_path
end

## ------------------------------------------------------------
## Run
## ------------------------------------------------------------


mkpath(OUT_DIR)

animate_prediction()
plot_probability_over_time()
println("\nDone.")