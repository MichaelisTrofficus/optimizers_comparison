from autograd import grad

import plotly.graph_objects as go

from surfaces.surfaces import hyperbolic_paraboloid
from surfaces.surfaces_plots import hyperbolic_paraboloid_fig

from optimizers.sgd import SGD
from optimizers.momentum import Momentum
from optimizers.nesterov import Nesterov
from optimizers.rmsprop import RMSProp
from optimizers.adam import Adam

# PLOTLY CONFIGURATION VARIABLES
PLOT_FIGURE = True

function_grad = grad(hyperbolic_paraboloid)
start_params = [-1.0, 0.001]

print(f"Start params: {start_params}")
print(f"Initial function value: {hyperbolic_paraboloid(start_params)}")
print("Starting Algorithm ...")

history_sgd = SGD(gradient=function_grad, params=start_params).get_history()
history_momentum = Momentum(gradient=function_grad, params=start_params).get_history()
history_nesterov = Nesterov(gradient=function_grad, params=start_params).get_history()
history_rmsprop = RMSProp(gradient=function_grad, params=start_params).get_history()
history_adam = Adam(gradient=function_grad, params=start_params).get_history()


if PLOT_FIGURE:

    xyz = hyperbolic_paraboloid_fig(200)
    x_sgd, y_sgd = history_sgd[:, 0], history_sgd[:, 1]
    x_momentum, y_momentum = history_momentum[:, 0], history_momentum[:, 1]
    x_nesterov, y_nesterov = history_nesterov[:, 0], history_nesterov[:, 1]
    x_rmsprop, y_rmsprop = history_rmsprop[:, 0], history_rmsprop[:, 1]
    x_adam, y_adam = history_adam[:, 0], history_adam[:, 1]

    fig = go.Figure(data=[go.Surface(
        x=xyz[0], y=xyz[1], z=xyz[2], colorscale="Greys",
        opacity=0.8, showscale=False)]*6
    )

    frames = [go.Frame(
        data=[
            go.Scatter3d(
                x=[x_sgd[k]],
                y=[y_sgd[k]],
                z=[hyperbolic_paraboloid([x_sgd[k], y_sgd[k]])],
                mode="markers",
                marker=dict(color="red", size=6, opacity=1),
                name="sgd"),
            go.Scatter3d(
                x=[x_momentum[k]],
                y=[y_momentum[k]],
                z=[hyperbolic_paraboloid([x_momentum[k], y_momentum[k]])],
                mode="markers",
                marker=dict(color="green", size=6, opacity=1),
                name="momentum"),
            go.Scatter3d(
                x=[x_nesterov[k]],
                y=[y_nesterov[k]],
                z=[hyperbolic_paraboloid([x_nesterov[k], y_nesterov[k]])],
                mode="markers",
                marker=dict(color="yellow", size=6, opacity=1),
                name="nesterov"),
            go.Scatter3d(
                x=[x_rmsprop[k]],
                y=[y_rmsprop[k]],
                z=[hyperbolic_paraboloid([x_rmsprop[k], y_rmsprop[k]])],
                mode="markers",
                marker=dict(color="pink", size=6, opacity=1),
                name="rmsprop"),
            go.Scatter3d(
                x=[x_adam[k]],
                y=[y_adam[k]],
                z=[hyperbolic_paraboloid([x_adam[k], y_adam[k]])],
                mode="markers",
                marker=dict(color="black", size=6, opacity=1),
                name="adam")
        ],
        name=f"frame{k}",
        traces=[1, 2, 3, 4, 5],
        ) for k in range(history_sgd.shape[0])]

    fig.update(frames=frames)
    fig.update_layout(updatemenus=[dict(type='buttons',
                                        buttons=[dict(label='Play',
                                                      method='animate',
                                                      args=[None,
                                                            dict(frame=dict(redraw=True,
                                                                            fromcurrent=True,
                                                                            mode="inmediate",
                                                                            duration=100))])],
                                       ),
                                   ]
                      )

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.6
    ))
    # fig.update_scenes(xaxis=dict(range=[-1.1, 1.1], title="X-axis", autorange=False),
    #                   yaxis=dict(range=[-1.1, 1.1], title="Y-axis", autorange=False),
    #                   zaxis=dict(range=[-1.1, 1.1], title="Z-axis", autorange=False))
    fig.show()
