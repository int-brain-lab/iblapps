"""
Wheel trace viewer.  Requires cv2

Example 1 - inspect trial 100 of a given session
    from dlc.wheel_dlc_viewer import Viewer
    eid = '77224050-7848-4680-ad3c-109d3bcd562c'
    v = Viewer(eid=eid, trial=100)

Example 2 - pick a random session to inspect
    from dlc.wheel_dlc_viewer import Viewer
    Viewer()

"""
import time
from datetime import date, timedelta
import random
import sys
import logging
from itertools import cycle
from more_itertools import chunked

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from one.api import ONE
import brainbox.behavior.wheel as wh
import ibllib.io.video as vidio


class Viewer:
    def __init__(self, eid=None, trial=None, camera='left', dlc_features=None, quick_load=True,
                 stream=True, t_win=3, one=None, start=True):
        """
        Plot the wheel trace alongside the video frames.  Below is list of key bindings:
        :key n: plot movements of next trial
        :key p: plot movements of previous trial
        :key r: plot movements of a random trial
        :key t: prompt for a trial number to plot
        :key l: toggle between legend for wheel and trial events
        :key space: pause/play frames
        :key left: move to previous frame
        :key right: move to next frame

        :param eid: uuid of experiment session to load
        :param trial: the trial id to plot
        :param camera: the camera position to load, options: 'left' (default), 'right', 'body'
        :param dlc_features: tuple of dlc features overlay onto frames
        :param quick_load: when true, move onset detection is performed on individual trials
        instead of entire session
        :param stream: when true, the video is streamed remotely instead of downloading
        :param t_win: the window in seconds over which to plot the wheel trace
        :param start: if False, the Viewer must be started by calling the `run` method
        :return: Viewer object
        """
        self._logger = logging.getLogger('ibllib')

        self.t_win = t_win  # Time window of wheel plot
        self.one = one or ONE()
        self.quick_load = quick_load

        # Input validation
        camera = vidio.assert_valid_label(camera)

        # If None, randomly pick a session to load
        if not eid:
            self._logger.info('Finding random session')
            eids = self.find_sessions(camera=camera, dlc=dlc_features is not None)
            eid = random.choice(eids)
            ref = self.one.eid2ref(eid, as_dict=False)
            self._logger.info('using session %s (%s)', eid, ref)
        else:
            eid = self.one.to_eid(eid)

        # Store complete session data: trials, timestamps, etc.
        ref = self.one.eid2ref(eid, parse=False)
        self._session_data = {'eid': eid, 'ref': ref, 'dlc': None}
        self._plot_data = {}  # Holds data specific to current plot, namely data for single trial

        # Download the DLC data if required
        self.dlc_features = dlc_features
        if dlc_features:
            self._session_data['dlc'] = self.get_dlc(camera, dlc_features)

        # These are for the dict returned by ONE
        trial_data = self.get_trial_data()
        total_trials = trial_data['intervals'].shape[0]
        trial = random.randint(0, total_trials) if trial is None else trial
        self._session_data['total_trials'] = total_trials
        self._session_data['trials'] = trial_data

        # Download the raw video for a given camera only
        if stream:
            self.video_path = vidio.url_from_eid(eid, camera, self.one)
        else:
            self.video_path, = self.download_raw_video(camera)

        cam_ts = self.one.load_dataset(eid, f'_ibl_{camera}Camera.times.npy')
        Fs = 1 / np.diff(cam_ts).mean()  # Approx. frequency of camera timestamps
        # Verify video frames and timestamps agree
        meta = vidio.get_video_meta(self.video_path)

        if meta['length'] != cam_ts.size:
            assert meta['length'] <= cam_ts.size, 'fewer camera timestamps than frames'
            msg = 'number of timestamps does not match number video file frames: '
            self._logger.warning(msg + '%i more timestamps than frames', cam_ts.size - meta['length'])

        assert Fs - meta['fps'] < 1, 'camera timestamps do not match reported frame rate'
        self._logger.info("Frame rate = %.0fHz", meta['fps'])
        # cam_ts = cam_ts[-count:]  # Remove extraneous timestamps
        self._session_data['camera_ts'] = cam_ts

        # Load wheel data
        self._session_data['wheel'] = self.one.load_object(eid, 'wheel')
        self._session_data['wheel_moves'] = self.one.load_object(eid, 'wheelMoves')

        # Plot the first frame in the upper subplot
        fig, axes = plt.subplots(nrows=2)
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)  # Disable defaults
        fig.canvas.mpl_connect('key_press_event', self.process_key)  # Connect our own key press fn

        self._plot_data['figure'] = fig
        self._plot_data['axes'] = axes
        self._trial_num = trial

        self.anim = animation.FuncAnimation(fig, self.animate, init_func=self.init_plot,
                                            frames=cycle(range(60)), interval=20, blit=False,
                                            repeat=True, cache_frame_data=False)
        self.anim.running = False
        self.trial_num = trial  # Set trial and prepare plot/frame data
        if start:
            self.run()

    def run(self):
        self._logger.debug('Running Viewer')
        plt.show()  # Start animation

    @property
    def trial_num(self):
        return self._trial_num

    @trial_num.setter
    def trial_num(self, trial):
        """
        Setter for the trial_num property.  Loads frames for trial, extracts onsets
        for trial and reinitializes plot
        :param trial: the trial number to select.  Must be > 0, <= len(trials['intervals'])
        :return: None
        """
        # Validate input: trial must be within range (1, total trials)
        trial = int(trial)
        total_trials = self._session_data['total_trials']
        if not 0 < trial <= total_trials:
            raise IndexError(
                'Trial number must be between 1 and {}'.format(total_trials))
        self._trial_num = trial
        sys.stdout.write('\rLoading trial ' + str(self._trial_num))

        # Our plot data, e.g. data that falls within trial
        frame_ids = self.frames_for_period(self._session_data['camera_ts'], trial - 1)
        try:
            data = {
                'frames': frame_ids,
                'camera_ts': self._session_data['camera_ts'][frame_ids],
                'frame_images': vidio.get_video_frames_preload(self.video_path, frame_ids)
            }
        except cv2.error as ex:
            # Log errors when out of memory; occurs when there are too many frames to load
            if ex.func.endswith('OutOfMemoryError'):
                self._logger.error(f'{ex.func}: {ex.err}')
                return
            else:
                raise ex
        except MemoryError:
            self._logger.error('Out of memory; please try loading another trial')
            return
        #  frame = get_video_frame(video_path, frames[0])

        on, off, ts, pos = self.extract_onsets_for_trial()
        data['moves'] = {'intervals': np.c_[on, off]}
        data['wheel'] = {'ts': ts, 'pos': pos}

        # Update title
        ref = '{date:s}_{sequence:s}_{subject:s}'.format(**self._session_data['ref'])
        if 'firstMovement_times' in self._session_data['trials']:
            first_move = self._session_data['trials']['firstMovement_times'][trial - 1]
            go_cue = self._session_data['trials']['goCue_times'][trial - 1]
            rt = (first_move - go_cue) * 1000
            title = '%s #%i rt = %.2f ms' % (ref, trial, rt)
        else:
            title = '%s #%i' % (ref, trial)

        self._plot_data['axes'][0].set_title(title)

        # Get the sample numbers for each onset and offset
        onoff_samps = np.c_[np.searchsorted(ts, on), np.searchsorted(ts, off)]
        data['moves']['onoff_samps'] = np.array(onoff_samps)
        # Points to split trace
        data['moves']['indicies'] = np.sort(np.hstack(onoff_samps))
        data['frame_num'] = 0
        data['figure'] = self._plot_data['figure']
        data['axes'] = self._plot_data['axes']
        if 'im' in self._plot_data.keys():
            # Copy over artists
            data['im'] = self._plot_data['im']
            data['ln'] = self._plot_data['ln']
            data['dlc'] = self._plot_data['dlc']

        # Stop running so we have to to reinitialize the plot after swapping out the plot data
        if self.anim.running:
            self.anim.running = False
            if self.anim:  # deals with issues on cleanup
                self.anim.event_source.stop()
        self._plot_data = data

    def find_sessions(self, camera='left', dlc=False):
        """
        Compile list of eids with required files, i.e. raw camera and wheel data
        :param dlc: search for sessions with DLC output
        :return: list of session eids
        """
        datasets = [f'{camera}Camera.raw', f'{camera}Camera.times', 'wheel.timestamps', 'wheel.position']
        if dlc:
            datasets.append(f'{camera}Camera.dlc')
        from_date = date.today() - timedelta(days=90)  # Narrow search for speed
        return self.one.search(date=[from_date, None], data=datasets)

    def download_raw_video(self, cameras=None):
        """
        Downloads the raw video from FlatIron or cache dir.  This allows you to download just one
        of the three videos
        :param cameras: the specific camera to load (i.e. 'left', 'right', or 'body') If None all
        three videos are downloaded.
        :return: the file path(s) of the raw videos
        """
        one = self.one
        eid = self._session_data['eid']
        datasets = one.list_datasets(eid, '*Camera.raw*')
        if cameras:
            cameras = [cameras] if isinstance(cameras, str) else cameras
            # Check cameras exist
            available = list(map(vidio.label_from_path, datasets))
            if missing := set(cameras) - set(available):
                raise ValueError(f'the following video(s) are not available: {missing}')
            datasets = [d for i, d in enumerate(datasets) if available[i] in cameras]

        return one.dowload_datasets(eid, datasets, download_only=True)

    def get_trial_data(self):
        """
        Obtain dict of trial data
        :return: dict of ALF trials object
        """
        return self.one.load_object(self._session_data['eid'], 'trials')

    def get_dlc(self, camera='left', features='all'):
        """
        Load the DLC data from file and discard features we don't need.
        :param camera: which camera dlc data to load ('left', 'right', 'body')
        :param features: filter columns on the given feature(s)
        :return: DLC DataFrame
        """
        dlc = self.one.load_dataset(self._session_data['eid'], f'_ibl_{camera}Camera.dlc')
        # Remove features we don't wish to plot
        if features != 'all':
            if isinstance(features, str):
                return dlc.filter(like=features, axis=1)
            else:
                features = '|'.join(map(lambda s: s + '_.+', features))
                return dlc.filter(regex=features, axis=1)
        else:
            return dlc

    def update_dlc_plot(self, frame_idx=0):
        """
        Update the coordinates and alpha values of DLC overlay markers
        :param frame_idx: the index in the trial's frame list
        :return: None
        """
        i = self._plot_data['frames'][frame_idx]
        dlc = self._session_data['dlc']
        # Loop through labels list range in threes, i.e. [0, 1, 2], [3, 4, 5]
        for j, ln in zip(chunked(range(len(dlc.columns)), 3), self._plot_data['dlc']):
            x, y, p = dlc.iloc[i, j]
            ln.set_offsets(np.c_[x, y])  # Update marker position
            ln.set_alpha(p)  # Update alpha based on likelihood

    def frames_for_period(self, cam_ts, start_time=None, end_time=None):
        """
        Load video frames between two events
        :param cam_ts: a camera.times numpy array
        :param start_time: a timestamp for the start of the period. If an int, the trial
        interval start at that index is used.  If None, period starts at first frame
        :param end_time: a timestamp for the end of the period. If an int, the trial
        interval end at that index is used.  If None, period ends at last frame, unless start_time
        is an int, in which case the trial interval at the start_time index is used
        :return: numpy bool mask the same size as cam_ts
        """
        if isinstance(start_time, int):
            trial_ends = self._session_data['trials']['intervals'][:, 1]
            trial_starts = self._session_data['trials']['intervals'][:, 0]
            end_time = trial_ends[start_time] if not end_time else trial_ends[end_time]
            start_time = trial_starts[start_time]
        else:
            if not start_time:
                start_time = cam_ts[0]
            if not end_time:
                end_time = cam_ts[-1]
        mask = np.logical_and(cam_ts >= start_time, cam_ts <= end_time)
        return np.where(mask)[0]

    def extract_onsets_for_trial(self):
        """
        Extracts the movement onsets and offsets for the current trial
        :return: tuple of onsets, offsets on, interpolated timestamps and interpolated positions
        """
        wheel = self._session_data['wheel']
        trials = self._session_data['trials']
        moves = self._session_data['wheel_moves']
        trial_idx = self.trial_num - 1  # Trials num starts at 1

        # Interpolate and get onsets
        pos, t = wh.interpolate_position(wheel['timestamps'], wheel['position'], freq=1000)
        # Get the positions and times between our trial start and the next trial start
        if self.quick_load or not self.trial_num:
            try:
                # End of previous trial to beginning of next
                t_mask = np.logical_and(t >= trials['intervals'][trial_idx - 1, 1],
                                        t <= trials['intervals'][trial_idx + 1, 0])
            except IndexError:  # We're on the last trial
                # End of previous trial to end of current
                t_mask = np.logical_and(t >= trials['intervals'][trial_idx - 1, 1],
                                        t <= trials['intervals'][trial_idx, 1])
        else:
            t_mask = np.ones_like(t, dtype=bool)
        wheel_ts = t[t_mask]
        wheel_pos = pos[t_mask]
        on_off = moves['intervals']
        on_off = on_off[np.logical_and(on_off[:, 0] >= wheel_ts[0], on_off[:, 0] < wheel_ts[-1])]
        on, off = np.hsplit(on_off, 2)
        return on, off, wheel_ts, wheel_pos

    def init_plot(self):
        """
        Plot the wheel data for the current trial
        :return: None
        """
        self._logger.debug('Initializing plot')
        data = self._plot_data
        trials = self._session_data['trials']
        trial_idx = self.trial_num - 1
        if 'im' in data:
            data['im'].set_data(data['frame_images'][0])
        else:
            data['im'] = data['axes'][0].imshow(data['frame_images'][0])

        # Plot DLC features
        dlc = self._session_data.get('dlc')
        dlc_paths = []  # DLC marker PathsCollections
        if dlc:  # DLC data loaded
            if data.get('dlc'):  # Already initialized
                self.update_dlc_plot()  # Update positions for first frame
            else:  # Plot new elements
                i = data['frames'][0]  # First frame
                cmap = ['b', 'g', 'r', 'c', 'm', 'y']
                # Each feature has three dimensions (x, y, likelihood)
                labels = chunked(range(len(dlc.columns)), 3)
                for j, colour in zip(labels, cycle(sorted(cmap * 2))):
                    x, y, p = dlc.iloc[i, j]
                    mkr = data['axes'][0].scatter(x, y, marker='+', c=colour, s=100, alpha=p)
                    dlc_paths.append(mkr)
        data['dlc'] = dlc_paths or None

        data['axes'][0].axis('off')

        indicies = data['moves']['indicies']
        on = data['moves']['intervals'][:, 0]
        off = data['moves']['intervals'][:, 1]
        onoff_samps = data['moves']['onoff_samps']
        wheel_pos = data['wheel']['pos']
        wheel_ts = data['wheel']['ts']
        cam_ts = data['camera_ts']

        # Plot the wheel position
        ax = data['axes'][1]
        ax.clear()
        ax.plot(on, wheel_pos[onoff_samps[:, 0]], 'go', label='move onset')
        ax.plot(off, wheel_pos[onoff_samps[:, 1]], 'bo', label='move offset')
        if 'firstMovement_times' in trials:
            first_move = trials['firstMovement_times'][trial_idx]
            if ~np.isnan(first_move):
                first_move_pos = wheel_pos[np.where(wheel_ts > first_move)[0][0]]
                ax.plot(first_move, first_move_pos, 'ro', label='first move')

        t_split = np.split(np.vstack((wheel_ts, wheel_pos)).T, indicies, axis=0)
        ax.add_collection(LineCollection(t_split[0::2], colors='k', label='wheel position'))
        ax.add_collection(LineCollection(t_split[1::2], colors='r', label='in movement'))  # Moving
        ax.set_ylabel('position / rad')
        wheel_legend = data['legends'][0] if 'legends' in data else plt.legend(title='Wheel')

        # Plot some trial events
        t1 = trials['intervals'][trial_idx, 0]
        t2 = trials['feedback_times'][trial_idx]
        t3 = trials['goCue_times'][trial_idx]
        t4 = trials['stimOn_times'][trial_idx]
        pos_rng = [wheel_pos.min(), wheel_pos.max()]  # The range for vertical lines on plot
        ax.vlines([t1, t2, t3, t4], pos_rng[0], pos_rng[1],
                  colors=['r', 'b', 'y', 'g'], linewidth=0.5)

        # Create legend for the trial events
        if 'legends' not in data:
            labels = ['trial start', 'feedback', 'stim on', 'go cue']
            custom_lines = [Line2D([0], [0], color='r', lw=.5),
                            Line2D([0], [0], color='b', lw=.5),
                            Line2D([0], [0], color='y', lw=.5),
                            Line2D([0], [0], color='g', lw=.5)]
            trial_legend = plt.legend(custom_lines, labels, title='Trial Events')
            trial_legend.set_visible(False)
            data['legends'] = (wheel_legend, trial_legend)
        else:
            trial_legend = data['legends'][1]

        # Add the legends to the plot
        ax.add_artist(wheel_legend)
        ax.add_artist(trial_legend)

        # Set limits and add frame marker
        ax.set_ylim(pos_rng)
        data['ln'] = ax.axvline(x=cam_ts[0], color='k')
        ax.set_xlim([cam_ts[0] - (self.t_win / 2), cam_ts[0] + (self.t_win / 2)])

        self._plot_data = data

        return data['im'], data['ln'], data['dlc']

    def animate(self, i):
        """
        Callback for figure animation.  Sets image data for current frame and moves pointer
        along axis
        :param i: unused; the current timestep of the calling method
        :return: None
        """
        t_start = time.time()
        data = self._plot_data
        if i < 0:
            self._plot_data['frame_num'] -= 1
            if self._plot_data['frame_num'] < 0:
                self._plot_data['frame_num'] = len(data['frame_images']) - 1
        else:
            self._plot_data['frame_num'] += 1
            if self._plot_data['frame_num'] >= len(data['frame_images']):
                self._plot_data['frame_num'] = 0
        i = self._plot_data['frame_num']  # NB: This is index for current trial's frame list
        # Print current frame number to terminal
        sys.stdout.flush()
        sys.stdout.write('\rFrame {} / {}'.format(i, len(data['frame_images'])))

        frame = data['frame_images'][i]
        t_x = data['camera_ts'][i]
        data['ln'].set_xdata([t_x, t_x])
        data['axes'][1].set_xlim([t_x - (self.t_win / 2), t_x + (self.t_win / 2)])
        data['im'].set_data(frame)
        if data['dlc']:
            self.update_dlc_plot(i)
        self._logger.debug('Render time: %.3f', time.time() - t_start)

        return data['im'], data['ln']

    def process_key(self, event):
        """
        Callback for key presses.
        :param event: a figure key_press_event
        :return: None
        """
        print(event.key)
        total_trials = self._session_data['total_trials']
        if event.key.isspace():
            if self.anim.running:
                self.anim.event_source.stop()
            else:
                self.anim.event_source.start()
            self.anim.running = ~self.anim.running
        elif event.key == 'right':
            if self.anim.running:
                self.anim.event_source.stop()
                self.anim.running = False
            self.animate(1)
            self._plot_data['figure'].canvas.draw()
        elif event.key == 'left':
            if self.anim.running:
                self.anim.event_source.stop()
                self.anim.running = False
            self.animate(-1)
            self._plot_data['figure'].canvas.draw()
        elif event.key == 'r':
            # Pick random trial
            self.trial_num = random.randint(0, total_trials)
            self.init_plot()
            self.anim.event_source.start()
            self.anim.running = True
        elif event.key == 't':
            if self.anim.running:
                self.anim.event_source.stop()
                self.anim.running = False
            # Select trial
            trial = input(f'\rInput a trial within range (1, {total_trials}): \n')
            if trial:
                self.trial_num = int(trial)
                self.init_plot()
            self.anim.event_source.start()
            self.anim.running = True
        elif event.key == 'n':
            # Next trial
            self.trial_num = self.trial_num + 1 if self.trial_num < total_trials else 1
            self.init_plot()
            self.anim.event_source.start()
            self.anim.running = True
        elif event.key == 'p':
            # Previous trial
            self.trial_num = self.trial_num - 1 if self.trial_num > 1 else total_trials
            self.init_plot()
            self.anim.event_source.start()
            self.anim.running = True
        elif event.key == 'l':
            # Toggle legend
            legends = self._plot_data['legends']
            for leg in legends:
                leg.set_visible(not leg.get_visible())
            if not self.anim.running:
                pass


if __name__ == "__main__":
    import textwrap
    import argparse

    docstring = """
    Viewer to see the wheel data along label the video and DLC.

    Below is list of key bindings:\n
        n         plot movements of next trial
        p         plot movements of previous trial
        r         plot movements of a random trial
        t         prompt for a trial number to plot
        l         toggle between for wheel and for trial events
        <space>   pause/play frames
        <left>    move to previous frame
        <right>   move to next frame
    """
    mpl.use('TKAgg')  # Different backend required for event callbacks in main

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(docstring))
    parser.add_argument('--eid', help='session uuid')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    parser.add_argument('--camera', default='left', help='the camera to load, choices are '
                                                         '"left" (default), "right" and "body"')
    parser.add_argument('-t', '--trial', type=int, help="trial number")
    parser.add_argument('--dlc', help='a comma separated list of DLC features to plot, or "all"')
    parser.add_argument('-w', '--window', type=int, default=3,
                        help="length of time window in wheel plot, default=3")
    args = vars(parser.parse_args())
    verbose = args.pop('verbose')

    dlc_features = args.pop('dlc')
    if dlc_features and dlc_features != 'all':
        dlc_features = dlc_features.split(',')
    args['t_win'] = args.pop('window')
    args['dlc_features'] = dlc_features

    v = Viewer(**args, start=False)
    if verbose:
        v._logger.setLevel(logging.DEBUG)
    # Start the viewer
    v.run()
