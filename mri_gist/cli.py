import click
from pathlib import Path
from mri_gist.utils.logging import setup_logger
from mri_gist.config import load_config

# CLI click test

# register
# segment
# separate
# convert
# pipeline
# serve webapp

@click.group()
@click.version_option(version='0.1.0')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """MRI-GIST: Comprehensive MRI Processing Toolkit"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logger(verbose)

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('template_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True)
@click.option('--method', type=click.Choice(['rigid', 'affine', 'syn']), default='syn')
@click.option('--threads', '-t', default=4, help='Number of threads')
def register(input_file, template_file, output, method, threads):
    """Register brain MRI to template space"""
    # Import here to avoid slow startup
    from mri_gist.registration.core import register_image
    
    click.echo(f"Registering {input_file} to {template_file}")
    register_image(
        moving=input_file,
        fixed=template_file,
        output=output,
        transform_type=method,
        num_threads=threads
    )
    click.secho(f"✓ Registration complete: {output}", fg='green')

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True)
@click.option('--robust', is_flag=True, help='Use robust mode')
@click.option('--parc', is_flag=True, help='Enable parcellation')
@click.option('--qc', type=click.Path(), help='Quality control output CSV')
def segment(input_file, output, robust, parc, qc):
    """Segment brain tissues using SynthSeg"""
    from mri_gist.segmentation.synthseg import run_synthseg
    
    run_synthseg(
        input_path=input_file,
        output_path=output,
        robust=robust,
        parcellation=parc,
        qc_path=qc
    )
    click.secho(f"✓ Segmentation complete: {output}", fg='green')

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-left', '-l', type=click.Path(), required=True)
@click.option('--output-right', '-r', type=click.Path(), required=True)
@click.option('--method', type=click.Choice(['antspy', 'flirt']), default='antspy')
def separate(input_file, output_left, output_right, method):
    """Separate brain into left and right hemispheres"""
    from mri_gist.detection.hemisphere import hemisphere_separation

    hemisphere_separation(
        input_path=input_file,
        left_output=output_left,
        right_output=output_right,
        method=method
    )
    click.secho(f"✓ Hemispheres separated", fg='green')

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True)
@click.option('--format', type=click.Choice(['nrrd', 'nii', 'nii.gz', 'vtk', 'stl', 'obj']), required=True)
@click.option('--clean', is_flag=True, help='Apply background cleaning (Otsu masking)')
def convert(input_file, output, format, clean):
    """Convert between medical imaging formats"""
    from mri_gist.conversion.formats import convert_format
    
    convert_format(
        input_path=input_file,
        output_path=output,
        target_format=format,
        clean_background=clean
    )
    click.secho(f"✓ Conversion complete: {output}", fg='green')

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--dry-run', is_flag=True, help='Show pipeline steps without running')
def pipeline(config_file, dry_run):
    """Run full processing pipeline from YAML config"""
    from mri_gist.pipeline import run_pipeline
    
    config = load_config(config_file)
    run_pipeline(config, dry_run=dry_run)

@cli.command()
@click.option('--port', default=8080, help='Port for web server')
@click.option('--host', default='localhost', help='Host address')
def serve(port, host):
    """Launch web-based visualization interface"""
    from mri_gist.visualization.server import start_server
    
    click.echo(f"Starting visualization server at http://{host}:{port}")
    start_server(host=host, port=port)

if __name__ == '__main__':
    cli()
