/**
 * Quick Settings — Alpine.js component.
 *
 * Tier 2 controls in the compose pane: aspect ratio,
 * image count, output format, seed.
 *
 * State is read by the Generate button when submitting to /api/generate.
 * All values initialize from $store.config defaults.
 */

function quickSettings() {
    return {
        aspectRatio: '',
        imageNumber: 2,
        outputFormat: 'png',
        seed: -1,
        randomSeed: true,
        defaultsApplied: false,

        init() {
            const applyDefaults = () => {
                if (this.defaultsApplied) return;
                this.defaultsApplied = true;
                const cfg = Alpine.store('config');
                this.aspectRatio = cfg.defaultAspectRatio || '';
                this.imageNumber = cfg.defaultImageNumber || 2;
                this.outputFormat = cfg.defaultOutputFormat || 'png';
                this.seed = -1;
                this.randomSeed = true;
                this._syncToStore();
            };

            if (Alpine.store('config').loaded) {
                applyDefaults();
            } else {
                this.$watch('$store.config.loaded', (loaded) => {
                    if (loaded) applyDefaults();
                });
            }

            // Keep shared store in sync whenever any quick setting changes.
            const watchedKeys = ['aspectRatio', 'imageNumber',
                                 'outputFormat', 'seed', 'randomSeed'];
            watchedKeys.forEach((key) => {
                this.$watch(key, () => this._syncToStore());
            });
        },

        /** Push current quick-settings values into $store.generation so
         *  the Generate button (outside this component) can read them. */
        _syncToStore() {
            const gen = Alpine.store('generation');
            gen.aspectRatio = this.aspectRatio;
            gen.imageNumber = this.imageNumber;
            gen.outputFormat = this.outputFormat;
            gen.seed = this.effectiveSeed;
        },

        get effectiveSeed() {
            return this.randomSeed ? -1 : this.seed;
        },

        updateImageNumber(value) {
            const parsed = Number.parseInt(value, 10);
            const max = Alpine.store('config').maxImageNumber || 32;
            this.imageNumber = Number.isFinite(parsed) ? Math.min(Math.max(parsed, 1), max) : 1;
        },

        toggleRandomSeed() {
            this.randomSeed = !this.randomSeed;
            if (this.randomSeed) {
                this.seed = -1;
            } else if (this.seed < 0) {
                this.seed = 0;
            }
        },
    };
}
