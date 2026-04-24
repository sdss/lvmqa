from multiprocessing import Pool
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
from pathlib import Path

pointings = pd.read_csv("/home/dr20_pointings_final_v1.2.0.csv")

def read_dap():
	base_dir = ('/data/sdss/lvm/analysis/1.2.0')
	files = []
	for row in pointings.itertuples(index=False):
		path = (
			Path(base_dir)
			/ str(row.tilegrp)
			/ str(row.tileid)
			/ str(row.mjd)
			/ f"{int(row.expnum):08d}"
			/ f"dap-rsp108-sn20-{int(row.expnum):08d}.dap.fits.gz"
		)
		if path.exists():
			files.append(path)
	files = sorted(set(files))
	print(f"Found {len(files)} files.")
	return files

# read_dap()

def read_ha_save(outputdir):
	df = pd.DataFrame(columns=['tileid', 'mjd', 'expnum', 'ra', 'dec', 'fiberid','ha_snr', 'ha_flux', 'ha_flux_err'])
	for row in pointings.itertuples(index=False):
		path = (
			Path('/data/sdss/lvm/analysis/1.2.0')
			/ str(row.tilegrp)
			/ str(row.tileid)
			/ str(row.mjd)
			/ f"{int(row.expnum):08d}"
			/ f"dap-rsp108-sn20-{int(row.expnum):08d}.dap.fits.gz"
		)
		if not path.exists():
			continue
		with fits.open(str(path), mode="readonly") as hdu:
			ha_flux = hdu[6].data["flux_Halpha_6562.85"] 
			ha_flux_err = hdu[6].data["e_flux_Halpha_6562.85"] 
			ha_snr = np.where(ha_flux_err > 0, ha_flux / ha_flux_err, np.nan) 
			fiberid = hdu[1].data['fiberid']
			ra = hdu[1].data['ra']
			dec = hdu[1].data['dec']
			tileid = row.tileid
			mjd = row.mjd
			expnum = row.expnum
			df_temp = pd.DataFrame({'tileid': tileid, 'mjd': mjd, 'expnum': expnum, 'ra': ra, 'dec': dec, 'fiberid': fiberid,'ha_snr': ha_snr, 'ha_flux': ha_flux, 'ha_flux_err': ha_flux_err})
			df = pd.concat([df, df_temp], ignore_index=True)
	df.to_parquet(outputdir)


def read_ha_np_snr(file=None):
	if file is None:
		files = read_dap()
	elif isinstance(file, (str, Path)):
		files = [Path(file)]
	else:
		files = [Path(f) for f in file]

	ha_snr_med = []
	ha_flux_med = []
	ha_flux_err_med = []
	for f in files:
		if not Path(f).exists():
			continue
		with fits.open(str(f), mode="readonly") as hdu:
			ha_flux = hdu[6].data["flux_Halpha_6562.85"] 
			ha_flux_err = hdu[6].data["e_flux_Halpha_6562.85"] 
			ha_snr = np.where(ha_flux_err > 0, ha_flux / ha_flux_err, np.nan) 
			ha_snr_med.append(np.nanmedian(ha_snr))
			index = np.argmin(np.abs(ha_snr - ha_snr_med[-1]))
			ha_flux_med.append(ha_flux[index])
			ha_flux_err_med.append(ha_flux_err[index])

	return ha_snr_med, ha_flux_med, ha_flux_err_med

# ------------------------ Plotting functions ------------------- #

def plot_ha_snr_compare():
    # ha snr compare
	qa_ha = pd.read_parquet('./ha_snr_dr20_v8_sci.parquet')
	dap_fiber = pd.read_parquet('./ha_np_dap.parquet')

	match_cols = ['tileid', 'mjd', 'expnum']
	qa_med = (
		qa_ha[match_cols + ['ha_snr_med', 'n_valid_spectra']]
		.rename(columns={'ha_snr_med': 'qa_ha_snr_med', 'n_valid_spectra': 'qa_n_valid'})
	)
	dap_med = (
		dap_fiber.assign(ha_snr=dap_fiber['ha_snr'].replace([np.inf, -np.inf], np.nan))
		.groupby(match_cols, as_index=False)
		.agg(dap_ha_snr_med=('ha_snr', 'median'), dap_n_valid=('ha_snr', 'count'))
	)
	ha_med = qa_med.merge(dap_med, on=match_cols, how='inner')
	ha_med = ha_med.replace([np.inf, -np.inf], np.nan).dropna(subset=['qa_ha_snr_med', 'dap_ha_snr_med'])


	fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
	sc = ax.scatter(
		ha_med['dap_ha_snr_med'],
		ha_med['qa_ha_snr_med'],
		c=ha_med['qa_n_valid'],
		s=35,
		alpha=0.8,
		cmap='viridis',
		edgecolors='none',
	)

	finite = ha_med[['dap_ha_snr_med', 'qa_ha_snr_med']].to_numpy().ravel()
	finite = finite[np.isfinite(finite)]
	lo, hi = np.nanpercentile(finite, [1, 99])
	pad = 0.05 * (hi - lo) if hi > lo else 1
	lo -= pad
	hi += pad
	ax.plot([lo, hi], [lo, hi], color='0.3', linestyle='--', linewidth=1, label='1:1')
	ax.set_xlim(lo, hi)
	ax.set_ylim(lo, hi)
	ax.set_aspect('equal', adjustable='box')
	ax.set_title('QA vs DAP Median H-alpha SNR')
	ax.set_xlabel('DAP median H-alpha SNR')
	ax.set_ylabel('QA median H-alpha SNR')
	ax.legend(loc='upper left')
	fig.colorbar(sc, ax=ax, label='QA valid spectra')
	# plt.savefig('./plots/qa_dap_med_v8.pdf', dpi=300)


def plot_ha_snr_hist():
	files = read_dap()
	np_snr, np_flux, np_flux_err = read_ha_np_snr(files)

	plt.close('all')
	fig, ax = plt.subplots(1, 1, figsize=(7, 5))

	ax.hist(np_snr, bins=60, color='tab:orange', alpha=0.8, label='DAP', range=(0,300))
	ax.set_title('Ha SNR Median')

	ax.set_xlabel('Ha SNR')
	ax.set_ylabel('Number of tiles')
	ax.axvline(np.median(np_snr), color='orange', linestyle='dashed', linewidth=1, label=f'Median DAP: {np.median(np_snr):.2f}')

	df = pd.read_parquet('./ha_snr_dr20_v8_sci.parquet')
	ax.hist(df['ha_snr_med'], bins=60, color='tab:green', alpha=0.8, label='QA fit', range=(0,300))
	ax.axvline(np.median(df['ha_snr_med']), color='green', linestyle='dashed', linewidth=1, label=f'Median: {np.median(df["ha_snr_med"]):.2f}')	

	ax.legend()

	# fig.savefig('./plots/ha_snr_stat_v8.pdf', dpi=300)
 
def plot_cont_med_hist():
	df = pd.read_parquet('./cont_snr_dr20_v2.parquet')
	fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15, 5), layout='constrained')
	global_range = (-3,3)

	ax[0].hist(df['b_snr_med'], bins=60, color='tab:cyan', alpha=0.8, range=global_range)
	ax[0].set_title('B band (5400-5500A)')
	ax[0].set_xlabel('Continuum SNR')
	ax[0].set_ylabel('Number of tiles')

	ax[1].hist(df['r_snr_med'], bins=60, color='tab:orange', alpha=0.8, range=global_range)
	ax[1].set_title('R band (6000-6100A)')
	ax[1].set_xlabel('Continuum SNR')
	ax[1].set_ylabel('Number of tiles')

	ax[2].hist(df['z_snr_med'], bins=60, color='tab:green', alpha=0.8, range=global_range)
	ax[2].set_title('Z band (8800-8900A)')
	ax[2].set_xlabel('Continuum SNR')
	ax[2].set_ylabel('Number of tiles')

	plot_list = ['b_snr_med', 'r_snr_med', 'z_snr_med']
	axi = ax.flatten()
	for i in range(3):
		median_val = np.median(df[plot_list[i]])
		axi[i].axvline(median_val, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.2f}')
		axi[i].legend(loc='upper left')
		n_high = (df[plot_list[i]] > global_range[1]).sum()
		n_low = (df[plot_list[i]] < global_range[0]).sum()
		axi[i].annotate(
			f'SNR> {global_range[1]}: {n_high}\nSNR< {global_range[0]}: {n_low}',
			xy=(0.98, 0.98),
			xycoords='axes fraction',
			ha='right',
			va='top',
			bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
		)
	
	# fig.savefig('./plots/continuum_snr_histogram_v2.pdf')
 
def plot_cont_specid():
	# spectrograph median continuum SNR comparison
	cont_fiber = pd.read_parquet("./cont_snr_dr20_v4_fibers.parquet")
	cont_fiber = cont_fiber.loc[
		(cont_fiber["targettype"] == "science")
		& cont_fiber["spectrographid"].isin([1, 2, 3])
	].copy()

	match_cols = ["tileid", "mjd", "expnum"]
	snr_cols = ["b_snr", "r_snr", "z_snr"]
	spec_medians = (
		cont_fiber.replace([np.inf, -np.inf], np.nan)
		.groupby(match_cols + ["spectrographid"], as_index=False)[snr_cols]
		.median()
	)

	band_specs = [
		("b_snr", "B continuum SNR"),
		("r_snr", "R continuum SNR"),
		("z_snr", "Z continuum SNR"),
	]
	spec_pairs = [(1, 2), (1, 3), (2, 3)]

	fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), layout="constrained")

	for row, (snr_col, band_label) in enumerate(band_specs):
		band_table = spec_medians.pivot_table(
			index=match_cols, columns="spectrographid", values=snr_col
		)
		band_values = band_table[[1, 2, 3]].to_numpy().ravel()
		band_values = band_values[np.isfinite(band_values)]
		lo, hi = np.nanpercentile(band_values, [1, 99])
		pad = 0.05 * (hi - lo) if hi > lo else 1
		lo -= pad
		hi += pad

		for col, (spec_x, spec_y) in enumerate(spec_pairs):
			ax = axes[row, col]
			pair_data = band_table[[spec_x, spec_y]].dropna()
			ax.scatter(
				pair_data[spec_x],
				pair_data[spec_y],
				s=24,
				alpha=0.75,
				color="tab:blue",
				edgecolors="none",
			)
			ax.set_xlim(lo, hi)
			ax.set_ylim(lo, hi)
			ax.set_aspect("equal", adjustable="box")
			ax.set_title(f"{band_label}: S{spec_x} vs S{spec_y}")
			ax.set_xlabel(f"Spectrograph {spec_x} median SNR")
			ax.set_ylabel(f"Spectrograph {spec_y} median SNR")
			ax.annotate(f"N = {len(pair_data)}", xy=(0.04, 0.94), xycoords="axes fraction", va="top")
			ax.plot([lo, hi], [lo, hi], color="0.3", linestyle="--", linewidth=1)


	# fig.savefig('./plots/continuum_snr_spec_compare.pdf', dpi=300)


if __name__ == "__main__":
	plot_ha_snr_hist()
	plot_cont_med_hist()
	plot_cont_specid()
	plot_ha_snr_compare()