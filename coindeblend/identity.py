import matplotlib.colors as mcl

paper_palette = [
    '#c6bcc0',
    '#1d456d',
    '#759792',
    '#ba9123',
    '#2f6b99',
    '#64532e',
    '#070c13',
    '#a3351a',
    '#0f3849',
    '#c66978',
    '#d5b56b',
    '#19252e',
    '#111b24',
    '#2a5650',
    '#24352b',
    '#162423',
    '#0f1c1b',
    '#1c181e',
    '#34241c',
]

gal1_cmap = mcl.LinearSegmentedColormap.from_list('paper_blue', ((1,1,1), '#1d456d'), N=256)
gal2_cmap = mcl.LinearSegmentedColormap.from_list('paper_brown', ((1,1,1), '#64532e'), N=256)
img_cmap = mcl.LinearSegmentedColormap.from_list('paper_BlBr', ['#1d456d', (0.8,0.8,0.8), '#64532e'])
