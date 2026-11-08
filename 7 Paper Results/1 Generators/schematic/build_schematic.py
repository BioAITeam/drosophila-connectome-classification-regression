#!/usr/bin/env python3
"""Export the editable connectome overview or rebuild it from archived assets."""
from pathlib import Path
import argparse
import shutil
import base64
import html
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path(__file__).with_name('connectome-learning-overview.drawio'))
parser.add_argument('--assets', type=Path, default=REPO / '5 Iris/5 Outputs/paper-schematic/assets')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rebuild', action='store_true', help='Rebuild diagram cells from the layout and archived assets')
args = parser.parse_args()
HERE = args.output.resolve()
if HERE.exists() and any(HERE.iterdir()):
    parser.error('Choose an empty output directory.')
HERE.mkdir(parents=True, exist_ok=True)
ASSETS = args.assets.resolve()
WIDTH, HEIGHT = 1600, 1268
BLACK, GRAY, BORDER = '#111111', '#5D6873', '#B9C2CA'
BLUE, LEARN_BORDER = '#1764AB', '#9B9000'
FRAME_BORDER = LEARN_BORDER
# Yellow denotes trainable quantities and supervised loss.
FIXED, LEARN = '#F6F8FA', '#F1E600'
mxfile = ET.Element('mxfile', host='app.diagrams.net', version='24.7.17', type='device')
diagram = ET.SubElement(mxfile, 'diagram', id='overview', name='Connectome learning overview')
model = ET.SubElement(diagram, 'mxGraphModel', dx=str(WIDTH), dy=str(HEIGHT), grid='0',
                      gridSize='10', guides='1', tooltips='1', connect='1', arrows='1',
                      fold='1', page='1', pageScale='1', pageWidth=str(WIDTH),
                      pageHeight=str(HEIGHT), math='0', shadow='0', background='#FFFFFF')
root = ET.SubElement(model, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', value='A - Reference framework', parent='0')
ET.SubElement(root, 'mxCell', id='2', value='B - Iris inference', parent='0')
layer = '1'
counter = 0
origins = {'1': (0, 0), '2': (0, 0)}


def cell_id():
    global counter
    counter += 1
    return f'cell-{counter:03d}'


def vertex(x, y, w, h, value='', style='', name=None):
    ox, oy = origins[layer]
    c = ET.SubElement(root, 'mxCell', id=name or cell_id(), value=value, style=style,
                      vertex='1', parent=layer)
    ET.SubElement(c, 'mxGeometry', x=str(x-ox), y=str(y-oy), width=str(w), height=str(h),
                  attrib={'as': 'geometry'})
    return c.get('id')


def box(x, y, w, h, fill=FIXED, stroke=BORDER, sw=1.5, name=None, rounded=False):
    return vertex(x, y, w, h, style=f'rounded={int(rounded)};absoluteArcSize=1;arcSize=16;'
                  f'whiteSpace=wrap;html=0;fillColor={fill};'
                  f'strokeColor={stroke};strokeWidth={sw};', name=name)


def dot(x, y, diameter=10):
    return vertex(x, y, diameter, diameter, style='ellipse;shape=ellipse;'
                  f'fillColor={GRAY};strokeColor=none;')


def text(x, y, w, h, value, size=20, bold=False, align='center', color=BLACK):
    return vertex(x, y, w, h, value, style='text;html=0;whiteSpace=wrap;'
                  f'fillColor=none;strokeColor=none;fontFamily=Arial;fontSize={size};'
                  f'fontColor={color};fontStyle={1 if bold else 0};align={align};'
                  'verticalAlign=middle;spacing=0;')


def image(x, y, w, h, path):
    payload = base64.b64encode(path.read_bytes()).decode('ascii')
    return vertex(x, y, w, h, style='shape=image;imageAspect=0;aspect=fixed;'
                  'fillColor=none;strokeColor=none;image=data:image/png,' + payload + ';')


def line(points, color=GRAY, width=2, arrow=True, dashed=False):
    ox, oy = origins[layer]
    points = [(x-ox, y-oy) for x, y in points]
    c = ET.SubElement(root, 'mxCell', id=cell_id(), value='', edge='1', parent=layer,
                      style=f'edgeStyle=none;rounded=0;html=0;strokeColor={color};'
                      f'strokeWidth={width};endArrow={"block" if arrow else "none"};'
                      f'endFill=1;endSize=9;dashed={1 if dashed else 0};dashPattern=7 5;')
    g = ET.SubElement(c, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
    for name, p in [('sourcePoint', points[0]), ('targetPoint', points[-1])]:
        ET.SubElement(g, 'mxPoint', x=str(p[0]), y=str(p[1]), attrib={'as': name})
    if len(points) > 2:
        arr = ET.SubElement(g, 'Array', attrib={'as': 'points'})
        for p in points[1:-1]:
            ET.SubElement(arr, 'mxPoint', x=str(p[0]), y=str(p[1]))


text(32, 20, 1350, 36, 'A   Learning within the anatomical connectome', 28, True, 'left')

# Two input modalities share the fixed interface to neurons in the graph.
box(32, 106, 355, 190, fill='#FFFFFF', stroke=FRAME_BORDER, rounded=True)
text(48, 118, 323, 32, 'Iris measurements', 24, True)
text(48, 158, 323, 28, 'Classification: 4 features', 21)
text(48, 195, 323, 26, 'Regression: 3 features', 21)
line([(48, 239), (371, 239)], color=BORDER, width=1, arrow=False)
text(48, 251, 323, 30, 'Scaled values | fixed assignment', 19)

box(32, 343, 355, 264, fill='#FFFFFF', stroke=FRAME_BORDER, rounded=True)
text(48, 353, 323, 32, 'BRAP-Net: Ki-67 images', 24, True)
image(53, 397, 74, 74, ASSETS/'histopathology.png')
line([(137, 434), (165, 434)], width=1.7)
text(174, 403, 195, 26, 'Frozen ViT', 22, True)
text(174, 440, 195, 26, 'Phikon-v2 / DINOv2', 19)
line([(48, 483), (371, 483)], color=BORDER, width=1, arrow=False)
text(44, 490, 331, 26, 'Two blocks for 730 input neurons', 19)
text(48, 521, 323, 26, 'Phikon-v2: 1,024 = 730 + 294', 20)
text(48, 550, 323, 26, 'DINOv2: 768 = 730 + 38', 20)
text(44, 577, 331, 26, 'Standardized | fixed assignment', 19)

# All input and output neurons belong to this anatomical graph boundary.
box(460, 106, 708, 494, fill='#FFFFFF', stroke=FRAME_BORDER, sw=1.5,
    name='anatomical-core', rounded=True)
origins['anatomical-core'] = (460, 106)
layer = 'anatomical-core'
text(484, 120, 660, 37, 'Anatomical connectome', 27, True)
text(484, 169, 660, 27, 'MaleCNS v1.0 | 1,323 neurons | 40,755 directed connections', 20)
box(486, 210, 656, 42, fill='#EEF4F9', stroke='#D8E2EB', rounded=True)
text(496, 217, 636, 28, '730 input neurons (R7/R8)', 22, True)
image(628, 275, 372, 155, ASSETS/'iris-step-00.png')
box(486, 440, 656, 97, fill='#F5F8FB', stroke='#D8E2EB', rounded=True)
text(496, 445, 636, 29, '8 output neurons within this graph', 22, True)
for center, name, count in [(596, 'MBON09', 4), (814, 'MBON27', 2), (1032, 'MBON05', 2)]:
    text(center-95, 476, 190, 26, f'{name} ({count})', 20)
    for j in range(count):
        dot(center - (count*16-6)/2 + j*16, 512, 10)
box(646, 553, 336, 36, fill=LEARN, stroke=LEARN_BORDER, sw=1.5,
    name='gain-parameters', rounded=True)
text(654, 557, 320, 28, 'Trainable synaptic gains', 22, True)
layer = '1'
line([(387, 267), (421, 267), (421, 231), (486, 231)])
line([(387, 590), (421, 590), (421, 267)], arrow=False)

# Outside the graph: numerical aggregation and prediction, without new neurons.
box(1232, 239, 336, 228, fill=FIXED, stroke=FRAME_BORDER,
    name='prediction-outputs', rounded=True)
text(1244, 262, 312, 27, 'Classification: Iris / Ki-67', 21, True)
text(1244, 304, 312, 25, 'Class probabilities (softmax)', 20)
line([(1248, 353), (1552, 353)], color=BORDER, width=1, arrow=False)
text(1244, 365, 312, 26, 'Regression: Iris', 21, True)
text(1244, 398, 312, 25, 'Difference of two means', 20)
text(1244, 431, 312, 24, 'Mean(MBON09) - Mean(MBON27)', 18)
line([(1142, 489), (1198, 489), (1198, 353), (1232, 353)])

# Gradient flow returns to internal gains, not to a separate output layer.
text(32, 652, 615, 31, 'Only synaptic gains are trained.', 23, True, 'left')
text(32, 690, 615, 52, 'Wiring, neuronal parameters and\ninput/output mappings remain fixed.', 21, align='left')
box(950, 636, 204, 49, fill=FIXED, stroke=FRAME_BORDER, rounded=True)
text(958, 648, 188, 26, 'True class or value', 20)
box(1232, 622, 336, 76, fill=LEARN, stroke=LEARN_BORDER, sw=1.5,
    name='supervised-loss', rounded=True)
text(1244, 644, 312, 32, 'Supervised loss', 23, True)
line([(1400, 467), (1400, 622)])
text(1410, 530.5, 143, 28, 'Prediction', 19, align='left')
line([(1154, 660), (1232, 660)])
line([(1400, 698), (1400, 736), (814, 736), (814, 589)],
     color=LEARN_BORDER, width=2.3, dashed=True)
text(856, 703, 350, 27, 'Backpropagation', 21, align='left')

# Six saved states include zero activity before the first recurrent update.
layer = '2'
line([(32, 776), (1568, 776)], color=BORDER, width=1.2, arrow=False)
text(32, 798, 1100, 36, 'B   Iris inference with trained gains', 28, True, 'left')
text(32, 845, 1170, 28, 'Sepal length: 5.4 cm | Sepal width: 3.9 cm | Petal length: 1.7 cm | Petal width: 0.4 cm', 21, align='left')
text(32, 884, 1536, 48,
     'At each step, neuronal activity is recalculated using its previous state, the inputs and the previous activity of connected neurons.\n'
     'The final prediction is computed after step 24. Synaptic gains remain fixed throughout.',
     20, align='left')
text(1226, 800, 342, 27, 'Signed model activity', 20)
# Same blue-neutral-orange scale as the original animation.
anchors = [(23,100,171), (182,188,196), (199,101,21)]
for i in range(100):
    f=i/99
    j=0 if f < .5 else 1
    a=2*f if j==0 else 2*f-1
    color='#'+''.join(f'{round((1-a)*anchors[j][k]+a*anchors[j+1][k]):02x}' for k in range(3))
    box(1254+i*2.9, 838, 3, 12, fill=color, stroke=color, sw=0)
text(1239, 854, 32, 24, '-1', 18)
text(1383, 854, 32, 24, '0', 18)
text(1527, 854, 32, 24, '+1', 18)

data=json.loads((ASSETS/'snapshots.json').read_text())
for k, snapshot in enumerate(data['snapshots']):
    x=32+260*k
    card_id = f'iris-update-{snapshot["step"]:02d}'
    box(x, 954, 236, 290, fill='#FFFFFF', stroke=BORDER, sw=1.2,
        name=card_id, rounded=True)
    origins[card_id] = (x, 954)
    layer = card_id
    text(x+12, 967, 212, 30, f'Update {snapshot["step"]}', 23, True)
    image(x+10, 1007, 216, 216*193/462, ASSETS/f'iris-step-{snapshot["step"]:02d}.png')
    for c,(name,p) in enumerate(zip(data['classes'],snapshot['probabilities'])):
        y=1112+40*c
        text(x+12,y,98,26,name,22,align='left')
        box(x+12,y+29,212,5,fill='#EDF1F4',stroke='#EDF1F4',sw=0)
        if p>0:
            box(x+12,y+29,212*p,5,fill=BLUE,stroke=BLUE,sw=0)
        text(x+148,y,76,26,f'{100*p:.2f}%',22,align='right')
    layer = '2'
    if k < len(data['snapshots'])-1:
        line([(x+240,1053),(x+256,1053)],color=GRAY,width=1.5)

drawio=HERE/'connectome-learning-overview.drawio'
if args.rebuild:
    ET.indent(mxfile)
    ET.ElementTree(mxfile).write(drawio,encoding='utf-8',xml_declaration=True)
else:
    shutil.copy2(args.source, drawio)


def export_saved_drawio(source):
    """Export the native cells used here. Text, rectangles, edges remain vector."""
    tree=ET.parse(source)
    ns='http://www.w3.org/2000/svg'
    ET.register_namespace('',ns)
    def tag(name):return '{'+ns+'}'+name
    svg=ET.Element(tag('svg'),width=str(WIDTH),height=str(HEIGHT),viewBox=f'0 0 {WIDTH} {HEIGHT}')
    ET.SubElement(svg,tag('rect'),x='0',y='0',width=str(WIDTH),height=str(HEIGHT),fill='white')
    cells=tree.findall('.//mxCell')
    by_id={c.get('id'):c for c in cells}
    def parent_origin(c):
        parent=by_id.get(c.get('parent'))
        if parent is None or parent.get('vertex')!='1':return 0,0
        px,py=parent_origin(parent)
        pg=parent.find('mxGeometry')
        return px+float(pg.get('x','0')),py+float(pg.get('y','0'))
    text_inventory=[]
    for c in cells:
        if c.get('vertex')!='1' and c.get('edge')!='1':continue
        style={}
        for s in c.get('style','').split(';'):
            if '=' in s:
                key,val=s.split('=',1);style[key]=val
        g=c.find('mxGeometry')
        ox,oy=parent_origin(c)
        if c.get('edge')=='1':
            def point(name):
                p=g.find(f'mxPoint[@as="{name}"]');return (float(p.get('x')),float(p.get('y')))
            pts=[point('sourcePoint')]+[(float(p.get('x')),float(p.get('y'))) for p in g.findall('Array/mxPoint')]+[point('targetPoint')]
            pts=[(x+ox,y+oy) for x,y in pts]
            color=style['strokeColor'];width=style['strokeWidth']
            attrs={'d':'M '+' L '.join(f'{x:g} {y:g}' for x,y in pts),'fill':'none','stroke':color,'stroke-width':width,'stroke-linejoin':'miter'}
            if style.get('dashed')=='1':attrs['stroke-dasharray']='7 5'
            ET.SubElement(svg,tag('path'),**attrs)
            if style['endArrow']!='none':
                a,b=pts[-2:];dx,dy=b[0]-a[0],b[1]-a[1];norm=math.hypot(dx,dy);ux,uy=dx/norm,dy/norm
                tip=b;rear=(b[0]-10*ux,b[1]-10*uy)
                poly=[tip,(rear[0]+4.5*uy,rear[1]-4.5*ux),(rear[0]-4.5*uy,rear[1]+4.5*ux)]
                ET.SubElement(svg,tag('polygon'),points=' '.join(f'{x:g},{y:g}' for x,y in poly),fill=color)
            continue
        x,y,w,h=[float(g.get(a,'0')) for a in ['x','y','width','height']]
        x,y=x+ox,y+oy
        if style.get('shape')=='image':
            uri=style['image'].replace('data:image/png,','data:image/png;base64,',1)
            el=ET.SubElement(svg,tag('image'),x=f'{x:g}',y=f'{y:g}',width=f'{w:g}',height=f'{h:g}',preserveAspectRatio='none')
            el.set('{http://www.w3.org/1999/xlink}href',uri)
        elif c.get('style','').startswith('text;'):
            fontsize=float(style['fontSize']);align=style['align'];lines=c.get('value','').split('\n');lh=fontsize*1.2
            tx=x if align=='left' else (x+w if align=='right' else x+w/2)
            anchor={'left':'start','right':'end','center':'middle'}[align]
            for i,t in enumerate(lines):
                ty=y+h/2+(i-(len(lines)-1)/2)*lh+fontsize*.35
                e=ET.SubElement(svg,tag('text'),x=f'{tx:g}',y=f'{ty:g}',fill=style['fontColor'],
                    attrib={'font-family':'Arial, sans-serif','font-size':str(fontsize),'font-weight':'700' if style['fontStyle']=='1' else '400','text-anchor':anchor})
                e.text=t
            text_inventory.append({'text':c.get('value',''),'x':x,'y':y,'width':w,'height':h,'font_size':fontsize})
        elif style.get('shape')=='ellipse':
            ET.SubElement(svg,tag('ellipse'),cx=f'{x+w/2:g}',cy=f'{y+h/2:g}',
                          rx=f'{w/2:g}',ry=f'{h/2:g}',fill=style['fillColor'],
                          stroke=style.get('strokeColor','none'))
        else:
            ET.SubElement(svg,tag('rect'),x=f'{x:g}',y=f'{y:g}',width=f'{w:g}',height=f'{h:g}',
                          rx='8' if style.get('rounded')=='1' else '0',
                          fill=style.get('fillColor','none'),stroke=style.get('strokeColor','none'),
                          attrib={'stroke-width':style.get('strokeWidth','1')})
    dest=source.with_suffix('.svg')
    ET.ElementTree(svg).write(dest,encoding='utf-8',xml_declaration=True)
    for fmt in ['pdf','png']:
        subprocess.run(['rsvg-convert','--format',fmt,'--output',str(source.with_suffix('.'+fmt)),str(dest)],check=True)
    (HERE/'text-inventory.json').write_text(json.dumps(text_inventory,indent=2)+'\n')
    return {'native_cells':len([c for c in cells if c.get('vertex')=='1' or c.get('edge')=='1']),
            'editable_text_cells':len(text_inventory),'embedded_images':sum('shape=image;' in c.get('style','') for c in cells)}


result=export_saved_drawio(drawio)
(HERE/'build-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
