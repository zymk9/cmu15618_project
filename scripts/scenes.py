#! /usr/bin/env python3 -B

import click, glob, os, sys, math, json, csv


@click.group()
def cli():
    pass


@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='json')
@click.option('--mode', '-m', default='path')
@click.option('--envname', '-e', default='')
def view(directory='mcguire', scene='*', format='json', mode='path', envname=''):
    modes = {
        'path': '--resolution 1280 --bounces 8 --clamp 10',
        'embree': '--resolution 1280 --bounces 8 --clamp 10 --embreebvh',
        'eyelight': '--resolution 1280 -t eyelight --bounces 8 --clamp 10',
        'eyelight-quick': '--resolution 1280 --samples 16 --sampler eyelight --bounces 8 --clamp 10'
    }
    options = modes[mode]
    envoptions = f'--envname {envname}' if envname else ''
    for dirname in sorted(glob.glob(f'{directory}/{format}/{scene}')):
        if not os.path.isdir(dirname): continue
        if '/_' in dirname: continue
        extraoptions = ''
        if os.path.exists(f'{dirname}/yscene_render.txt'):
            with open(f'{dirname}/yscene_render.txt') as f:
                extraoptions = f.read().strip()
        for filename in sorted(glob.glob(f'{dirname}/*.{format}')):
            if format == 'pbrt':
                with open(filename) as f:
                    if 'WorldBegin' not in f.read(): continue
            cmd = f'../yocto-gl/bin/yscene view {options} {extraoptions} {envoptions} --scene {filename}'
            print(cmd, file=sys.stderr)
            os.system(cmd)


@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='json')
@click.option('--mode', '-m', default='path')
def render(directory='mcguire', scene='*', format='json', mode='path'):
    modes = {
        'path': '--samples 64 --resolution 1280 --bounces 8 --clamp 10',
        'path-face': '--samples 256 --resolution 1280 --bounces 8 --clamp 10',
        'embree': '--samples 256 --resolution 1280 --bounces 8 --clamp 10 --embreebvh',
        'eyelight': '--samples 16 --resolution 1280 --bounces 8 --clamp 10 --sampler eyelight',
        'embree-face': '--samples 1024 --resolution 1280 --bounces 8 --clamp 10 --embreebvh',
        'final': '--samples 4096 --resolution 1280 --bounces 8 --clamp 10 --embreebvh',
    }
    options = modes[mode]
    outformat = 'png' if 'eyelight' in mode else 'hdr'
    outprefix = 'eyelight' if 'eyelight' in mode else 'images'
    for dirname in sorted(glob.glob(f'{directory}/{format}/{scene}')):
        if not os.path.isdir(dirname): continue
        if '/_' in dirname: continue
        extracams = []
        if 'sanmiguel' in dirname: extracams = ['camera2', 'camera3']
        if 'island' in dirname: extracams = ["beachCam", "birdseyeCam", "dunesACam", "grassCam", "palmsCam", "rootsCam", "shotCam"]
        if 'landscape' in dirname: extracams = ['camera2', 'camera3', 'camera4']
        extraoptions = ''
        if os.path.exists(f'{dirname}/yscene_render.txt'):
            with open(f'{dirname}/yscene_render.txt') as f:
                extraoptions = f.read().strip()
        for filename in sorted(glob.glob(f'{dirname}/*.{format}')):
            if format == 'pbrt':
                with open(filename) as f:
                    if 'WorldBegin' not in f.read(): continue
            basename = os.path.basename(filename).replace(f'.{format}', '')
            os.system(f'mkdir -p {directory}/{outprefix}-{format}')
            imagename = f'{directory}/{outprefix}-{format}/{basename}.{outformat}'
            cmd = f'../yocto-gl/bin/yscene render --output {imagename} {options} {extraoptions} {filename}'
            print(cmd, file=sys.stderr)
            os.system(cmd)
            for idx, cam in enumerate(extracams, 1):
                imagename = f'{directory}/{outprefix}-{format}/{basename}-c{idx}.{outformat}'
                cmd = f'../yocto-gl/bin/yscene render --output {imagename} --camera {cam} {options} {extraoptions} --scene {filename}'
                print(cmd, file=sys.stderr)
                os.system(cmd)


@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='json')
@click.option('--mode', '-m', default='default')
def info(directory='mcguire', scene='*', format='json', mode='default'):
    modes = {
        'default': '',
        'validate': '--validate'
    }
    options = modes[mode]
    for dirname in sorted(glob.glob(f'{directory}/{format}/{scene}')):
        if not os.path.isdir(dirname): continue
        if '/_' in dirname: continue
        extraoptions = ''
        if os.path.exists(f'{dirname}/yscene_render.txt'):
            with open(f'{dirname}/yscene_render.txt') as f:
                extraoptions = f.read().strip()
        for filename in sorted(glob.glob(f'{dirname}/*.{format}')):
            if format == 'pbrt':
                with open(filename) as f:
                    if 'WorldBegin' not in f.read(): continue
            cmd = f'../yocto-gl/bin/yscene info {options} {extraoptions} --scene {filename}'
            print(cmd, file=sys.stderr)
            os.system(cmd)

@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='json')
@click.option('--mode', '-m', default='default')
def validate(directory='mcguire', scene='*', format='json', mode='default'):
    modes = {
        'default': ''
    }
    options = modes[mode]
    schema = '../yocto-gl/scripts/scene.schema.json'
    for dirname in sorted(glob.glob(f'{directory}/{format}/{scene}')):
        if not os.path.isdir(dirname): continue
        if '/_' in dirname: continue
        extraoptions = ''
        for filename in sorted(glob.glob(f'{dirname}/*.{format}')):
            cmd = f'../yocto-gl/scripts/validate-scene.py {schema} {filename} {options}'
            print(cmd, file=sys.stderr)
            os.system(cmd)


@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='json')
@click.option('--mode', '-m', default='linear')
def tonemap(directory='mcguire', scene='*', format='json', mode='filmic'):
    modes = {
        'linear': '',
        'filmic': '--filmic --exposure 0.5'
    }
    options = modes[mode]
    outformat = 'png'
    outprefix = 'images'
    from PIL import Image, ImageFont, ImageDraw
    fontname1 = '~/Library/Fonts/FiraSansCondensed-Regular.ttf'
    fontname2 = '~/Library/Fonts/FiraSansCondensed-Regular.ttf'
    font1 = ImageFont.truetype(fontname1, 30)
    font2 = ImageFont.truetype(fontname2, 18)
    for filename in sorted(
            glob.glob(f'{directory}/{outprefix}-{format}/{scene}.hdr') +
            glob.glob(f'{directory}/{outprefix}-{format}/{scene}.exr')):
        imagename = filename.replace(f'.exr', f'.{outformat}').replace(
            f'.hdr', f'.{outformat}')
        cmd = f'../yocto-gl/bin/yimage convert --output {imagename} {options} --image {filename}'
        print(cmd, file=sys.stderr)
        os.system(cmd)
        img = Image.open(imagename)
        w, h = img.size
        draw = ImageDraw.Draw(img)
        tw, _ = draw.textsize("Yocto/GL", font=font1)
        draw.rectangle([w - 8, h - 32 - 8, w - 8 - 8 - tw, h - 8], (0, 0, 0))
        draw.text((w - 8 - 4, h - 26 - 8 - 4), "Yocto/GL", (255, 255, 255), font=font1, anchor='rt')
        if directory in ['bitterli', 'disney', 'mcguire', 'pbrt3', 'yocto', 'heads', 'blender', 'fabio']:
            authorfilename = filename.replace('images-json/', f'{format}/').replace(
                '-fr.', '.').replace('-hr.', '.').replace('-c1.', '.').replace(
                    '-c2.', '.').replace('-c3.', '.').replace('-c4.', '.').replace(
                        '-c5.', '.').replace('-c6.', '.').replace('-c7.', '.').replace(
                            '.hdr', '') + '/AUTHOR.txt'
            print(authorfilename)
            with open(authorfilename) as f:
                text = f.read().strip()
            tw, _ = draw.textsize(text, font=font2)
            draw.rectangle([8, h - 26 - 8, 8 + 8 + tw, h - 8], (0, 0, 0))
            draw.text((8 + 4, h - 20 - 8 - 4), text, (255, 255, 255), font=font2)
        img.save(imagename)


@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='json')
@click.option('--mode', '-m', default='jpg')
def gallery(directory='mcguire', scene='*', format='json', mode='filmic'):
    modes = {
        'jpg': ''
    }
    options = modes[mode]
    inprefix = 'images'
    outformat = 'jpg'
    outprefix = 'gallery'
    os.system(f'mkdir -p {directory}/{outprefix}-{format}')
    from PIL import Image
    for filename in sorted(glob.glob(f'{directory}/{inprefix}-{format}/{scene}.png')):
        imagename = filename.replace(f'{inprefix}-', f'{outprefix}-').replace('.png',f'.{outformat}')
        print(filename, file=sys.stderr)
        img = Image.open(filename)
        rgb_img = img.convert('RGB')
        rgb_img.save(imagename)


@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='obj')
@click.option('--clean/--no-clean', '-C', default=False)
def sync_images(directory='mcguire',
                scene='*',
                format='obj',
                clean=True):
    for dirname in sorted(glob.glob(f'{directory}/{format}/{scene}')):
        if not os.path.isdir(dirname): continue
        if '/_' in dirname: continue
        for filename in sorted(glob.glob(f'{dirname}/*.{format}')):
            if format == 'pbrt':
                with open(filename) as f:
                    if 'WorldBegin' not in f.read(): continue
            basename = os.path.basename(filename).replace(f'.{format}', '')
            os.system(f'mkdir -p {directory}/images-{format}')
            imagename = f'{directory}/images-{format}/ytrace-{mode}-{basename}.*'
            if clean:
                cmd = f'rm {dirname}/*.png'
                print(cmd, file=sys.stderr)
                os.system(cmd)
                cmd = f'rm {dirname}/*.hdr'
                print(cmd, file=sys.stderr)
                os.system(cmd)
            cmd = f'cp {imagename} {dirname}'
            print(cmd, file=sys.stderr)
            os.system(cmd)


@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='obj')
@click.option('--outformat', '-F', default='json')
@click.option('--mode', '-m', default='default')
@click.option('--clean/--no-clean', '-C', default=False)
def convert(directory='mcguire',
            scene='*',
            format='obj',
            outformat="json",
            mode='path',
            clean=True):
    modes = {
        'default': '',
    }
    options = modes[mode]
    for dirname in sorted(glob.glob(f'{directory}/source/{scene}')):
        if not os.path.isdir(dirname): continue
        if '/_' in dirname: continue
        copyright_options = ''
        if os.path.exists(f'{dirname}/AUTHOR.txt'):
            with open(f'{dirname}/AUTHOR.txt') as f:
                copyright = f.read().strip().replace('"', '')
            copyright_options += f'--copyright "{copyright}"'
        outdirname = dirname.replace(f'/source/', f'/{outformat}/')
        if clean: os.system(f'rm -rf {outdirname}')
        os.system(f'mkdir -p {outdirname}')
        for auxname in ['AUTHOR.txt', 'LICENSE.txt', 'LINKS.txt', 'README.txt', 'yscene_render.txt']:
            if os.path.exists(f'{dirname}/{auxname}'):
                os.system(f'cp {dirname}/{auxname} {outdirname}/')
        for filename in sorted(glob.glob(f'{dirname}/*.{format}')):
            if format == 'pbrt':
                with open(filename) as f:
                    if 'WorldBegin' not in f.read(): continue
            outname = filename.replace(f'/source/', f'/{outformat}/').replace(
                f'.{format}', f'.{outformat}')
            cmd = f'../yocto-gl/bin/yscene convert --output {outname} {options} {filename} {copyright_options}'
            print(cmd, file=sys.stderr)
            os.system(cmd)


@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='obj')
@click.option('--outformat', '-F', default='json')
@click.option('--mode', '-m', default='default')
def copyright(directory='mcguire',
              scene='*',
              format='obj',
              outformat="json",
              mode='default'):
    modes = {
        'default': '',
    }
    options = modes[mode]
    for dirname in sorted(glob.glob(f'{directory}/source/{scene}')):
        if not os.path.isdir(dirname): continue
        if '/_' in dirname: continue
        outdirname = dirname.replace(f'/source/', f'/{outformat}/')
        os.system(f'mkdir -p {outdirname}')
        if os.path.exists(f'{dirname}/AUTHOR.txt'):
            os.system(f'cp {dirname}/AUTHOR.txt {outdirname}/')
        if os.path.exists(f'{dirname}/LICENSE.txt'):
            os.system(f'cp {dirname}/LICENSE.txt {outdirname}/')
        if os.path.exists(f'{dirname}/LINKS.txt'):
            os.system(f'cp {dirname}/LINKS.txt {outdirname}/')


@cli.command()
@click.option('--directory', '-d', default='yuksel')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='hair')
@click.option('--outformat', '-F', default='ply')
@click.option('--mode', '-m', default='default')
@click.option('--clean-models/--no-clean-models', '-C', default=False)
def convert_hair(directory='yuksel',
                 scene='*',
                 format='hair',
                 outformat="ply",
                 mode='path',
                 clean_models=True):
    modes = {'default': ''}
    options = modes[mode]
    for dirname in sorted(glob.glob(f'{directory}/{scene}')):
        if not os.path.isdir(dirname): continue
        if '/_' in dirname: continue
        if 'ecosys' in dirname and outformat == 'obj': continue
        if 'landscape' in dirname and outformat == 'obj': continue
        if 'fractal' in dirname and outformat == 'obj': continue
        if 'pavilion' in dirname and outformat == 'obj': continue
        for filename in sorted(glob.glob(f'{dirname}/{format}/*.{format}')):
            outname = filename.replace(f'/{format}/', f'/json/').replace(
                f'.{format}', f'.{outformat}')
            filedir = os.path.dirname(filename)
            cmd = f'../yocto-gl/bin/ymshproc -o {outname} {options} {filename}'
            print(cmd, file=sys.stderr)
            os.system(cmd)


@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='obj')
@click.option('--mode', '-m', default='default')
def backup(directory='mcguire', scene='*', format='obj', mode='default'):
    modes = {
        'default': '-r -X -q',
    }
    options = modes[mode]
    for dirname in sorted(glob.glob(f'{directory}/{format}/{scene}')):
        if not os.path.isdir(dirname): continue
        if '/_' in dirname: continue
        outdir = f'{directory}/backup-{format}'
        basedir = f'{directory}/{format}'
        os.system(f'mkdir -p {outdir}')
        dirname = dirname.replace(basedir + '/', '')
        outname = dirname + '.zip'
        os.system(f'rm {outdir}/{outname}')
        cmd = f'cd {basedir}; zip {options} {outname} {dirname}; mv {outname} ../../{outdir}/'
        print(cmd)
        os.system(cmd)


@cli.command()
@click.option('--directory', '-d', default='procedurals')
@click.option('--mode', '-m', default='skies')
@click.option('--clean/--no-clean', '-C', default=False)
def make_procedurals(directory='procedurals', mode='skies', clean=False):
    if mode == 'skies':
        dirname = f'{directory}/hdr/textures'
        os.system(f'mkdir -p {dirname}')
        angles = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90]
        for name in ['sky', 'sun']:
            for angle in angles:
                jsonname = f'{dirname}/_proc.json'
                outname = f'{dirname}/{name}-{angle:02}.hdr'
                js = {
                    'type': 'sky',
                    'width': 2048,
                    'height': 1024,
                    'sun_angle': math.radians(angle),
                    'has_sun': 'sun' in name,
                    'turbidity': 3
                }
                with open(jsonname, 'w') as f:
                    json.dump(js, f, indent=2)
                cmd = f'../yocto-gl/bin/yimproc -o {outname} {jsonname}'
                print(cmd, file=sys.stderr)
                os.system(cmd)
                os.system(f'rm {jsonname}')
    else:
        print('unknown mode')


@cli.command()
def sync():
    os.system(
        "rsync -avcm --delete --include '*/' --include '*.zip' --include '*.tgz' --include '*.pdf' --exclude='*' ./ ../yocto-scenes"
    )
    # os.system('rsync -avc --delete ./ ../yocto-scenes')


@cli.command()
@click.option('--directory', '-d', default='pbrt-v3-scenes')
@click.option('--scene', '-s', default='*')
def pbrtparse(directory='pbrt-v3-scenes', scene='*'):
    broken_scenes = [
        'bunny-fur/f3-15.pbrt',
        "dambreak/dambreak0.pbrt",
        "hair/curly-hair.pbrt",
        "contemporary-bathroom/contemporary-bathroom.pbrt",
        "head/head.pbrt",
        "ecosys/ecosys.pbrt",
        "sanmiguel/sanmiguel.pbrt",
        "sssdragon/dragon_50.pbrt",
        "white-room/whiteroom-daytime.pbrt",
    ]
    scenes = [
        'barcelona-pavilion/pavilion-day.pbrt',
        'barcelona-pavilion/pavilion-night.pbrt',
        'bathroom/bathroom.pbrt',
        'bmw-m6/bmw-m6.pbrt',
        'breakfast/breakfast.pbrt',
        'buddha-fractal/buddha-fractal.pbrt',
        'bunny-fur/f3-15.pbrt',
        'caustic-glass/glass.pbrt',
        "chopper-titan/chopper-titan.pbrt",
        "cloud/cloud.pbrt",
        "coffee-splash/splash.pbrt",
        "contemporary-bathroom/contemporary-bathroom.pbrt",
        "crown/crown.pbrt",
        "dambreak/dambreak0.pbrt",
        "dragon/f8-4a.pbrt",
        "ecosys/ecosys.pbrt",
        "ganesha/ganesha.pbrt",
        "hair/curly-hair.pbrt",
        "hair/sphere-hairblock.pbr",
        "head/head.pbrt",
        "killeroos/killeroo-simple.pbrt",
        "landscape/view-0.pbrt",
        "lte-orb/lte-orb-silver.pbrt",
        "measure-one/frame85.pbrt",
        "pbrt-book/book.pbrt",
        "sanmiguel/sanmiguel.pbrt",
        "simple/dof-dragons.pbrt",
        "smoke-plume/plume-184.pbrt",
        "sportscar/sportscar.pbrt",
        "sssdragon/dragon_50.pbrt",
        "structuresynth/microcity.pbrt",
        "transparent-machines/frame542.pbrt",
        "tt/tt.pbrt",
        "veach-bidir/bidir.pbrt",
        "veach-mis/mis.pbrt",
        "villa/villa-daylight.pbrt",
        "volume-caustic/caustic.pbrt",
        "vw-van/vw-van.pbrt",
        "white-room/whiteroom-daytime.pbrt",
        "yeahright/yeahright.pbrt",
    ]
    # for filename in scenes:
    #     if scene != '*' and not filename.startswith(f'{scene}/'): continue
    #     cmd = f'../yocto-gl/bin/yitrace {filename}'
    #     print(cmd, file=sys.stderr)
    #     os.system(cmd)
    for filename in scenes:
        if scene != '*' and not filename.startswith(f'{scene}/'): continue
        cmd = f'../yocto-gl/bin/yitrace {directory}/{filename}'
        print(cmd, file=sys.stderr)
        os.system(cmd)


@cli.command()
@click.option('--directory', '-d', default='mcguire')
@click.option('--scene', '-s', default='*')
@click.option('--format', '-f', default='json')
@click.option('--outformat', '-F', default='csv')
@click.option('--mode', '-m', default='default')
def stats(directory='mcguire',
          scene='*',
          format='json',
          outformat="csv",
          mode='default'):
    stats = []
    keys = [
        'name', 'cameras', 'environments', 'shapes', 'subdivs', 'textures',
        'stextures'
    ]
    for dirname in sorted(glob.glob(f'{directory}/{format}/{scene}')):
        if not os.path.isdir(dirname): continue
        if '/_' in dirname: continue
        for filename in sorted(glob.glob(f'{dirname}/*.{format}')):
            with open(filename) as f:
                scene = json.load(f)
            stat = {}
            stat['name'] = filename.partition('/')[2].partition('.')[0]
            stat['cameras'] = len(
                scene['cameras']) if 'cameras' in scene else 0
            stat['environments'] = len(
                scene['environments']) if 'environments' in scene else 0
            stat['shapes'] = len(scene['shapes']) if 'shapes' in scene else 0
            stat['subdivs'] = len(
                scene['subdivs']) if 'subdivs' in scene else 0
            textures = {}
            for shape in scene['shapes']:
                for key, value in shape.items():
                    if '_tex' not in key: continue
                    if value not in textures: textures[value] = 0
                    textures[value] += 1
            stat['textures'] = len(textures)
            stat['stextures'] = sum(count for _, count in textures.items())
            stats += [stat]
    os.system(f'mkdir -p {directory}/_stats-{format}')
    with open(f'{directory}/_stats-{format}/stats.{outformat}',
              'w',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for stat in stats:
            writer.writerow([stat[key] for key in keys])

@cli.command()
@click.option('--directory', '-d', default='mcguire')
def fix_objx(directory='mcguire'):
    for filename in glob.glob(directory + "/source/*/*.objx"):
        newname = filename.replace('.objx', '.obx')
        obx = ''
        with open(filename) as f:
            for line in f:
                if line.startswith('c'):
                    tokens = line.split()
                    for i in range(len(tokens)):
                        if i in [0, 1, 2]: continue
                        tokens[i] = float(tokens[i])
                    obx += 'newCam {}\n'.format(tokens[1])
                    obx += '  Ca {}\n'.format(round(tokens[3] / tokens[4], 3))
                    obx += '  Cl {}\n'.format(round(tokens[5] * 0.036 / tokens[3],3))
                    obx += '  Ct {} {} {} {} {} {} 0 1 0\n'.format(round(tokens[17], 2), round(tokens[18], 2), round(tokens[19], 2), round(tokens[17] - tokens[14] * tokens[6], 2), round(tokens[18] - tokens[15] * tokens[6], 2), round(tokens[19] - tokens[16] * tokens[6], 2))
                    obx += '\n';
                if line.startswith('e'):
                    tokens = line.split()
                    for i in range(len(tokens)):
                        if i in [0, 1, 5]: continue
                        tokens[i] = float(tokens[i])
                    obx += 'newEnv {}\n'.format(tokens[1])
                    obx += '  Ee {} {} {}\n'.format(round(tokens[2], 1), round(tokens[3], 1), round(tokens[4], 1))
                    if tokens[5] != '""': obx += '  map_Ee {}\n'.format(tokens[5])
                    obx += '  Et {} {} {} {} {} {} 0 1 0\n'.format(round(tokens[15], 2), round(tokens[16], 2), round(tokens[17], 2), round(tokens[15] + tokens[12], 2), round(tokens[16] + tokens[13], 2), round(tokens[17] + tokens[14], 2))
                    obx += '\n';
        with open(newname, 'wt') as f:
            f.write(obx)

cli()
