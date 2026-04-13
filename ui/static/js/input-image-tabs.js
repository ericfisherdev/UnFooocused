/* UNF-64 — Input Image tabs Alpine component.
 *
 * Scope: minimal base container for the three input-image sub-tabs
 * (Upscale or Variation, Image Prompt, Inpaint or Outpaint). The inpaint
 * tab exposes file upload + a canvas-based brush tool that paints white
 * strokes onto a transparent mask, then emits { image, mask } data URLs
 * via the `inpaintChanged` event so the generate handler can forward it
 * to /api/generate.
 */
document.addEventListener("alpine:init", () => {
    const TAB_ORDER = ["uov", "ip", "inpaint"];

    Alpine.data("inputImageTabs", () => ({
        activeTab: "inpaint",
        uovImage: null,
        inpaintImage: null,
        inpaintMode: "default",
        outpaintSelections: [],
        inpaintAdditionalPrompt: "",
        brushSize: 32,
        _drawing: false,
        _lastX: 0,
        _lastY: 0,
        _maskCtx: null,
        _imgCtx: null,
        _baseCanvas: null,
        _sourceImage: null,
        _pointerId: null,

        init() {
            this.$watch("inpaintImage", (value) => {
                if (value) {
                    this.$nextTick(() => this._loadImageIntoCanvas(value));
                }
            });
        },

        // WCAG tab pattern: arrow keys cycle, Home/End jump, focus follows.
        onTabKeydown(event) {
            const key = event.key;
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(key)) return;
            event.preventDefault();
            const current = TAB_ORDER.indexOf(this.activeTab);
            let next = current;
            if (key === "ArrowLeft") {
                next = (current - 1 + TAB_ORDER.length) % TAB_ORDER.length;
            } else if (key === "ArrowRight") {
                next = (current + 1) % TAB_ORDER.length;
            } else if (key === "Home") {
                next = 0;
            } else if (key === "End") {
                next = TAB_ORDER.length - 1;
            }
            this.activeTab = TAB_ORDER[next];
            this.$nextTick(() => {
                const btn = document.getElementById(`input-image-tab-${this.activeTab}-btn`);
                btn?.focus();
            });
        },

        onUovUpload(event) {
            const file = event.target.files?.[0];
            if (!file) return;
            this._readFileAsDataUrl(file).then((dataUrl) => {
                this.uovImage = dataUrl;
                this._dispatchUov();
            });
        },

        onInpaintUpload(event) {
            const file = event.target.files?.[0];
            if (!file) return;
            this._readFileAsDataUrl(file).then((dataUrl) => {
                this.inpaintImage = dataUrl;
            });
        },

        _readFileAsDataUrl(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = () => reject(reader.error);
                reader.readAsDataURL(file);
            });
        },

        _loadImageIntoCanvas(dataUrl) {
            const canvas = this.$refs.inpaintCanvas;
            if (!canvas) return;
            const img = new Image();
            img.onload = () => {
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0);
                this._imgCtx = ctx;
                // Cache decoded source so _redrawComposite draws synchronously
                // and avoids per-stroke Image.onload races.
                this._sourceImage = img;

                // Off-screen mask canvas — transparent by default, paint white.
                const mask = document.createElement("canvas");
                mask.width = canvas.width;
                mask.height = canvas.height;
                this._maskCtx = mask.getContext("2d");
                this._baseCanvas = mask;
                this._dispatchInpaint();
            };
            img.src = dataUrl;
        },

        _canvasCoords(event) {
            const canvas = this.$refs.inpaintCanvas;
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {
                x: (event.clientX - rect.left) * scaleX,
                y: (event.clientY - rect.top) * scaleY,
            };
        },

        startStroke(event) {
            if (!this._maskCtx) return;
            const canvas = this.$refs.inpaintCanvas;
            if (typeof event.pointerId === "number") {
                canvas?.setPointerCapture?.(event.pointerId);
                this._pointerId = event.pointerId;
            }
            const { x, y } = this._canvasCoords(event);
            this._drawing = true;
            this._lastX = x;
            this._lastY = y;
            this._paintPoint(x, y);
        },

        drawStroke(event) {
            if (!this._drawing || !this._maskCtx) return;
            if (typeof this._pointerId === "number" && event.pointerId !== this._pointerId) return;
            const { x, y } = this._canvasCoords(event);
            this._paintLine(this._lastX, this._lastY, x, y);
            this._lastX = x;
            this._lastY = y;
        },

        endStroke() {
            if (!this._drawing) return;
            this._drawing = false;
            const canvas = this.$refs.inpaintCanvas;
            if (typeof this._pointerId === "number") {
                canvas?.releasePointerCapture?.(this._pointerId);
                this._pointerId = null;
            }
            this._dispatchInpaint();
        },

        clearMask() {
            if (!this._maskCtx || !this._baseCanvas) return;
            this._maskCtx.clearRect(0, 0, this._baseCanvas.width, this._baseCanvas.height);
            this._redrawComposite();
            this._dispatchInpaint();
        },

        _paintPoint(x, y) {
            this._maskCtx.fillStyle = "#FFFFFF";
            this._maskCtx.beginPath();
            this._maskCtx.arc(x, y, this.brushSize / 2, 0, Math.PI * 2);
            this._maskCtx.fill();
            this._redrawComposite();
        },

        _paintLine(x0, y0, x1, y1) {
            const ctx = this._maskCtx;
            ctx.strokeStyle = "#FFFFFF";
            ctx.lineWidth = this.brushSize;
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();
            this._redrawComposite();
        },

        _redrawComposite() {
            const canvas = this.$refs.inpaintCanvas;
            if (!canvas || !this._baseCanvas || !this._sourceImage) return;
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(this._sourceImage, 0, 0);
            ctx.globalAlpha = 0.5;
            ctx.drawImage(this._baseCanvas, 0, 0);
            ctx.globalAlpha = 1.0;
        },

        _dispatchInpaint() {
            if (!this._baseCanvas) return;
            const payload = {
                image: this.inpaintImage,
                mask: this._baseCanvas.toDataURL("image/png"),
            };
            window.dispatchEvent(new CustomEvent("inpaint-changed", { detail: payload }));
        },

        _dispatchUov() {
            window.dispatchEvent(
                new CustomEvent("uov-changed", { detail: { image: this.uovImage } })
            );
        },
    }));
});
