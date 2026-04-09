import click
from pathlib import Path
from mri_gist.utils.logging import setup_logger
from mri_gist.config import load_config
from mri_gist.fractal.labels import get_default_lut_path

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

@cli.command(name='msp-align')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='./msp_output',
              help='Output directory for results')
@click.option('--interp-order', type=int, default=1,
              help='Interpolation order (0=nearest for labels, 1=linear)')
@click.option('--no-split', is_flag=True, help='Skip hemisphere separation')
@click.option('--no-diagnostics', is_flag=True, help='Skip PNG diagnostic outputs')
def msp_align(input_file, output_dir, interp_order, no_split, no_diagnostics):
    """Align midsagittal plane via 3D midway registration (flip + T^{1/2})"""
    from mri_gist.registration.midway import midway_align, hemisphere_split, save_diagnostics

    stem = Path(input_file).name.replace('.nii.gz', '').replace('.nii', '')
    result = midway_align(input_file, interp_order=interp_order)

    hemi = None
    if not no_split:
        hemi = hemisphere_split(
            result["aligned_data"], result["aligned_img"].affine, result["lr_axis"]
        )
        ratio = hemi["left_nonzero"] / max(hemi["right_nonzero"], 1)
        click.echo(f"  Hemisphere balance: L/R = {ratio:.3f}")

    if not no_diagnostics:
        written = save_diagnostics(result, hemi, output_dir, stem)
        click.echo(f"  Wrote {len(written)} files to {output_dir}/")
    else:
        # Save at least the aligned volume
        from pathlib import Path as P
        out = P(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        import nibabel as _nib
        _nib.save(result["aligned_img"], str(out / f"{stem}_midway_aligned.nii.gz"))

    click.secho(f"✓ MSP alignment complete: {stem}", fg='green')

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


@cli.group()
def fractal():
    """Fractal dimension workflows for segmentation volumes"""


def _summarize_fractal_results(dataframe):
    whole_brain_count = int((dataframe['scope'] == 'whole_brain').sum())
    label_count = int((dataframe['scope'] == 'label').sum())
    measurement_count = len(dataframe)

    if label_count:
        return (
            f"Saved {measurement_count} fractal measurements "
            f"({whole_brain_count} whole-brain, {label_count} label-wise)"
        )

    return f"Saved {measurement_count} whole-brain fractal measurement(s)"


def _summarize_label_recap(dataframe, preview_count=3):
    label_names = [
        str(name)
        for name in dataframe.loc[dataframe['scope'] == 'label', 'label_name'].dropna().unique().tolist()
    ]
    if not label_names:
        return None

    preview = ", ".join(label_names[:preview_count])
    remaining = len(label_names) - preview_count
    if remaining > 0:
        return f"Labels measured: {preview}, +{remaining} more"
    return f"Labels measured: {preview}"


@fractal.command(name='compute')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True, help='Output CSV path')
@click.option('--lut', type=click.Path(exists=True), help='Label lookup table for per-label outputs; defaults to the bundled FreeSurfer LUT')
@click.option('--per-label', is_flag=True, help='Compute per-label fractal dimensions in addition to whole-brain FD')
@click.option('--metadata', type=click.Path(exists=True), help='Optional CSV/TSV to merge onto results')
@click.option('--merge-key', 'merge_keys', multiple=True, default=['participant_id', 'session_id'], help='Merge key for metadata joins; may be repeated')
def fractal_compute(input_file, output, lut, per_label, metadata, merge_keys):
    """Compute fractal dimension for one segmentation volume"""
    from mri_gist.fractal import compute_segmentation_table, merge_metadata_table, write_results_csv

    click.echo(f"Starting fractal computation for {Path(input_file).name}")
    lut_path = lut if lut else str(get_default_lut_path())
    dataframe = compute_segmentation_table(
        input_file,
        lut_path=lut_path,
        per_label=per_label,
        show_progress=True,
    )
    if metadata:
        click.echo(f"Merging metadata from {Path(metadata).name}")
        dataframe = merge_metadata_table(dataframe, metadata, merge_keys=merge_keys)

    output_path = write_results_csv(dataframe, output)
    click.echo(_summarize_fractal_results(dataframe))
    label_recap = _summarize_label_recap(dataframe)
    if label_recap:
        click.echo(label_recap)
    click.secho(f"✓ Fractal results saved: {output_path}", fg='green')


@fractal.command(name='batch')
@click.argument('input_pattern', type=str)
@click.option('--output', '-o', type=click.Path(), required=True, help='Output CSV path')
@click.option('--lut', type=click.Path(exists=True), help='Label lookup table for per-label outputs; defaults to the bundled FreeSurfer LUT')
@click.option('--per-label', is_flag=True, help='Compute per-label fractal dimensions in addition to whole-brain FD')
@click.option('--metadata', type=click.Path(exists=True), help='Optional CSV/TSV to merge onto results')
@click.option('--merge-key', 'merge_keys', multiple=True, default=['participant_id', 'session_id'], help='Merge key for metadata joins; may be repeated')
def fractal_batch(input_pattern, output, lut, per_label, metadata, merge_keys):
    """Compute fractal dimension for all files matching a glob pattern"""
    from mri_gist.fractal import compute_batch_table, write_results_csv

    click.echo(f"Starting fractal batch for pattern: {input_pattern}")
    lut_path = lut if lut else str(get_default_lut_path())
    dataframe = compute_batch_table(
        input_pattern,
        lut_path=lut_path,
        per_label=per_label,
        metadata_path=metadata,
        merge_keys=merge_keys,
        show_progress=True,
    )

    output_path = write_results_csv(dataframe, output)
    volume_count = dataframe['source_file'].nunique()
    click.echo(f"Processed {volume_count} segmentation volume(s)")
    click.echo(_summarize_fractal_results(dataframe))
    label_recap = _summarize_label_recap(dataframe)
    if label_recap:
        click.echo(label_recap)
    click.secho(f"✓ Fractal batch results saved: {output_path}", fg='green')


@fractal.command(name='masks')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), required=True, help='Directory for generated binary masks')
@click.option('--lut', type=click.Path(exists=True), help='Label lookup table used to name mask files; defaults to the bundled FreeSurfer LUT')
@click.option('--overwrite', is_flag=True, help='Overwrite existing mask files')
def fractal_masks(input_file, output_dir, lut, overwrite):
    """Generate binary masks for each non-zero label in a segmentation volume"""
    from mri_gist.fractal import generate_binary_masks

    lut_path = lut if lut else str(get_default_lut_path())
    written_files = generate_binary_masks(
        input_file,
        output_dir,
        lut_path=lut_path,
        overwrite=overwrite,
    )
    click.echo(f"Generated {len(written_files)} mask files")
    click.secho(f"✓ Masks written to: {output_dir}", fg='green')

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
