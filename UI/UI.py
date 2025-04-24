import dash
from dash import dcc, html
import plotly.graph_objs as go
from collections import deque
import random

app = dash.Dash(__name__)

# 初始化数据
data_length = 100  # 显示的总数据长度
x = deque(maxlen=data_length)
y_red = deque(maxlen=data_length)
y_green = deque(maxlen=data_length)
y_blue = deque(maxlen=data_length)
y_marked = deque(maxlen=data_length)

# 随机初始化 x 轴和 y 轴数据
for i in range(data_length):
    x.append(i)
    y_red.append(random.uniform(0, 1))
    y_green.append(random.uniform(0, 1))
    y_blue.append(random.uniform(0, 1))
    y_marked.append(1)  # 默认标记为 True

app.layout = html.Div([
    dcc.Graph(id='original-graph'),  # 原始数据图
    dcc.Graph(id='marked-graph'),    # 标注数据图
    dcc.Interval(id='interval-component', interval=1000)  # 每秒更新
])

@app.callback(
    [dash.dependencies.Output('original-graph', 'figure'),
     dash.dependencies.Output('marked-graph', 'figure')],
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    try:
        # 每秒生成 10 个随机数据点
        new_data_red = [random.uniform(0, 1) for _ in range(10)]
        new_data_green = [random.uniform(0, 1) for _ in range(10)]
        new_data_blue = [random.uniform(0, 1) for _ in range(10)]
        # 模拟模型检测，判断整个区间是否为 False
        is_false = random.choice([True, False])  # 模拟模型对整个区间的判断

        # 更新原始数据
        start_x = x[-1] + 1 if x else 0
        for i in range(10):  # 每次更新 10 个点
            x.append(start_x + i)  # 更新 x 轴
            y_red.append(new_data_red[i])
            y_green.append(new_data_green[i])
            y_blue.append(new_data_blue[i])
            y_marked.append(0 if is_false else 1)  # 如果区间为 False，标记所有点为 0

        # 原始数据图
        original_figure = {
            'data': [
                go.Scatter(x=list(x), y=list(y_red), mode='lines', name='红色通道', line=dict(color='red')),
                go.Scatter(x=list(x), y=list(y_green), mode='lines', name='绿色通道', line=dict(color='green')),
                go.Scatter(x=list(x), y=list(y_blue), mode='lines', name='蓝色通道', line=dict(color='blue'))
            ],
            'layout': go.Layout(
                title='原始数据',
                xaxis=dict(title='时间'),
                yaxis=dict(title='值', range=[0, 1])
            )
        }

        # 标注数据图
        shapes = []
        for i in range(0, len(x), 10):  # 每 10 个点为一个区间
            # 确保区间不越界
            x0 = x[i]
            x1 = x[i + 10 - 1] if i + 10 <= len(x) else x[-1]
            # 检查区间是否为 False
            if all(mark == 0 for mark in list(y_marked)[i:i + 10]):
                shapes.append({
                    'type': 'rect',
                    'xref': 'x',
                    'yref': 'paper',
                    'x0': x0,
                    'x1': x1,
                    'y0': 0,
                    'y1': 1,
                    'fillcolor': 'rgba(0, 255, 0, 0.2)',  # 红色背景，透明度 0.2
                    'line': {'width': 0}
                })

        marked_figure = {
            'data': [
                go.Scatter(x=list(x), y=list(y_red), mode='lines', name='红色通道', line=dict(color='red')),
                go.Scatter(x=list(x), y=list(y_green), mode='lines', name='绿色通道', line=dict(color='green')),
                go.Scatter(x=list(x), y=list(y_blue), mode='lines', name='蓝色通道', line=dict(color='blue'))
            ],
            'layout': go.Layout(
                title='标注数据',
                xaxis=dict(title='时间'),
                yaxis=dict(title='值', range=[0, 1]),
                shapes=shapes  # 添加背景标记
            )
        }

        return original_figure, marked_figure

    except Exception as e:
        return {}, {}

if __name__ == '__main__':
    app.run(debug=True)