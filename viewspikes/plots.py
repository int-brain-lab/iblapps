# import modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pyqtgraph as pg

import ibllib.atlas as atlas
from neuropixel import SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.plots import wiggle, color_cycle

brain_atlas = atlas.AllenAtlas()
# Instantiate brain atlas and one


def show_psd(data, fs, ax=None):
    psd = np.zeros((data.shape[0], 129))
    for tr in np.arange(data.shape[0]):
        f, psd[tr, :] = scipy.signal.welch(data[tr, :], fs=fs)

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(f, 10 * np.log10(psd.T), color='gray', alpha=0.1)
    ax.plot(f, 10 * np.log10(np.mean(psd, axis=0).T), color='red')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB rel V/Hz)')
    ax.set_ylim(-150, -110)
    ax.set_xlim(0, fs / 2)
    plt.show()


def plot_insertion(pid, one=None):
    # Find eid of interest
    assert one
    insertion = one.alyx.rest('insertions', 'list', id=pid)[0]
    probe_label = insertion['name']
    eid = insertion['session']

    # Load in channels.localCoordinates dataset type
    chn_coords = one.load(eid, dataset_types=['channels.localCoordinates'])[0]
    depths = chn_coords[:, 1]

    # Find the ephys aligned trajectory for eid probe combination
    trajs = one.alyx.rest('trajectories', 'list', session=eid, probe=probe_label)
    #provenance=,
    traj_aligned = next(filter(lambda x: x['provenance'] == 'Ephys aligned histology track', trajs), None)
    if traj_aligned is None:
        raise NotImplementedError(f"Plots only aligned insertions so far - TODO")
    else:
        plot_alignment(insertion, traj_aligned, -1)
    # Extract all alignments from the json field of object
    # Load in the initial user xyz_picks obtained from track traccing


def plot_alignment(insertion, traj, ind=None):
    depths = SITES_COORDINATES[:, 1]
    xyz_picks = np.array(insertion['json']['xyz_picks']) / 1e6

    alignments = traj['json'].copy()
    k = list(alignments.keys())[-1]  # if only I had a Walrus available !
    alignments = {k: alignments[k]}
    # Create a figure and arrange using gridspec
    widths = [1, 2.5]
    heights = [1] * len(alignments)
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axis = plt.subplots(len(alignments), 2, constrained_layout=True,
                             gridspec_kw=gs_kw, figsize=(8, 9))

    # Iterate over all alignments for trajectory
    # 1. Plot brain regions that channel pass through
    # 2. Plot coronal slice along trajectory with location of channels shown as red points
    # 3. Save results for each alignment into a dict - channels
    channels = {}
    for iK, key in enumerate(alignments):

        # Location of reference lines used for alignmnet
        feature = np.array(alignments[key][0])
        track = np.array(alignments[key][1])
        chn_coords = SITES_COORDINATES
        # Instantiate EphysAlignment object
        ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track, feature_prev=feature)

        # Find xyz location of all channels
        xyz_channels = ephysalign.get_channel_locations(feature, track)
        # Find brain region that each channel is located in
        brain_regions = ephysalign.get_brain_locations(xyz_channels)
        # Add extra keys to store all useful information as one bunch object
        brain_regions['xyz'] = xyz_channels
        brain_regions['lateral'] = chn_coords[:, 0]
        brain_regions['axial'] = chn_coords[:, 1]

        # Store brain regions result in channels dict with same key as in alignment
        channel_info = {key: brain_regions}
        channels.update(channel_info)

        # For plotting -> extract the boundaries of the brain regions, as well as CCF label and colour
        region, region_label, region_colour, _ = ephysalign.get_histology_regions(xyz_channels, depths)

        # Make plot that shows the brain regions that channels pass through
        ax_regions = fig.axes[iK * 2]
        for reg, col in zip(region, region_colour):
            height = np.abs(reg[1] - reg[0])
            bottom = reg[0]
            color = col / 255
            ax_regions.bar(x=0.5, height=height, width=1, color=color, bottom=reg[0], edgecolor='w')
        ax_regions.set_yticks(region_label[:, 0].astype(int))
        ax_regions.yaxis.set_tick_params(labelsize=8)
        ax_regions.get_xaxis().set_visible(False)
        ax_regions.set_yticklabels(region_label[:, 1])
        ax_regions.spines['right'].set_visible(False)
        ax_regions.spines['top'].set_visible(False)
        ax_regions.spines['bottom'].set_visible(False)
        ax_regions.hlines([0, 3840], *ax_regions.get_xlim(), linestyles='dashed', linewidth=3,
                          colors='k')
        # ax_regions.plot(np.ones(channel_depths_track.shape), channel_depths_track, '*r')

        # Make plot that shows coronal slice that trajectory passes through with location of channels
        # shown in red
        ax_slice = fig.axes[iK * 2 + 1]
        brain_atlas.plot_tilted_slice(xyz_channels, axis=1, ax=ax_slice)
        ax_slice.plot(xyz_channels[:, 0] * 1e6, xyz_channels[:, 2] * 1e6, 'r*')
        ax_slice.title.set_text(insertion['id'] + '\n' + str(key))

    # Make sure the plot displays
    plt.show()


def overlay_spikes(self, spikes, clusters, channels, rgb=None, label='default', symbol='x', size=8):
    first_sample = self.ctrl.model.t0 / self.ctrl.model.si
    last_sample = first_sample + self.ctrl.model.ns

    ifirst, ilast = np.searchsorted(spikes['samples'], [first_sample, last_sample])
    tspi = spikes['samples'][ifirst:ilast].astype(np.float64) * self.ctrl.model.si
    xspi = channels['rawInd'][clusters['channels'][spikes['clusters'][ifirst:ilast]]]

    n_side_by_side = int(self.ctrl.model.ntr / 384)
    print(n_side_by_side)
    if n_side_by_side > 1:
        addx = (np.zeros([1, xspi.size]) + np.array([self.ctrl.model.ntr * r for r in range(n_side_by_side)])[:, np.newaxis]).flatten()
        xspi = np.tile(xspi, n_side_by_side) + addx
        yspi = np.tile(tspi, n_side_by_side)
    if self.ctrl.model.taxis == 1:
        self.ctrl.add_scatter(xspi, tspi, label=label)
    else:
        self.ctrl.add_scatter(tspi, xspi, label=label)
    sc = self.layers[label]['layer']
    sc.setSize(size)
    sc.setSymbol(symbol)
    # sc.setPen(pg.mkPen((0, 255, 0, 155), width=1))
    if rgb is None:
        rgbs = [list((rgb * 255).astype(np.uint8)) for rgb in color_cycle(spikes['clusters'][ifirst:ilast])]
        sc.setBrush([pg.mkBrush(rgb) for rgb in rgbs])
        sc.setPen([pg.mkPen(rgb) for rgb in rgbs])
    else:
        sc.setBrush(pg.mkBrush(rgb))
        sc.setPen(pg.mkPen(rgb))
    return sc, tspi, xspi

    # sc.setData(x=xspi, y=tspi, brush=pg.mkBrush((255, 0, 0)))
    def callback(sc, points, evt):
        NTR = 12
        NS = 128
        qxy = self.imageItem_seismic.mapFromScene(evt.scenePos())
        # tr, _, _, s, _ = self.ctrl.cursor2timetraceamp(qxy)
        itr, itime = self.ctrl.cursor2ind(qxy)
        print(itr, itime)
        h = self.ctrl.model.header
        x = self.ctrl.model.data
        trsel = np.arange(np.max([0, itr - NTR]), np.min([self.ctrl.model.ntr, itr + NTR]))
        ordre = np.lexsort((h['x'][trsel], h['y'][trsel]))
        trsel = trsel[ordre]
        w = x[trsel, int(itime) - NS:int(itime) + NS]
        wiggle(-w.T * 10000, fs=1 / self.ctrl.model.si, t0=(itime - NS) * self.ctrl.model.si)
        # hw = {k:h[k][trsel] for k in h}

    # s.sigMouseClicked.disconnect()
    sc.sigClicked.connect(callback)
    # self.ctrl.remove_all_layers()
