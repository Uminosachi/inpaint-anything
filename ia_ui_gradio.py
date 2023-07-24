import os

import gradio as gr

GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse


def webpath(fn):
    web_path = os.path.realpath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'


def javascript_html():
    script_path = os.path.join(os.path.dirname(__file__), "javascript", "inpaint-anything.js")
    head = f'<script type="text/javascript" src="{webpath(script_path)}"></script>\n'

    return head


def reload_javascript():
    js = javascript_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
