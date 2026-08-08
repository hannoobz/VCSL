# Usage:
#   make extract   isc  full
#   make extract   isc  pred
#   make extract   isc  gt
#   make simmap    dino full
#   make simmapviz dino full
#   make vta       isc  pred TN
#   make vta       isc  gt   SPD
#   make eval      isc  gt   SPD

# BACKBONE  = isc | dino
# MODE      = full | pred | gt      (full = uncropped, pred/gt = pip_only crops)
# METHOD    = TN | SPD              (only needed for vta / eval)

ROOT       ?= ./output
DATASET    ?= ./dataset
PAIR_FILE  ?= $(DATASET)/pair_file_test.csv
SPLIT      ?= test

BACKBONE := $(word 2,$(MAKECMDGOALS))
MODE     := $(word 3,$(MAKECMDGOALS))
METHOD   := $(word 4,$(MAKECMDGOALS))

$(BACKBONE) $(MODE) $(METHOD):
	@:

ifeq ($(MODE),full)
  CROP_MODE  := full
  MODE_PATH  := full
  YOLO_ARGS  :=
else ifeq ($(MODE),pred)
  CROP_MODE  := pip_only
  MODE_PATH  := pip_only/pred
  YOLO_ARGS  := --yolo-root $(DATASET)/labels_pred --yolo-separate-folder --label-set pred
else ifeq ($(MODE),gt)
  CROP_MODE  := pip_only
  MODE_PATH  := pip_only/gt
  YOLO_ARGS  := --yolo-root $(DATASET)/labels --yolo-separate-folder --label-set gt
endif

SPD_MODEL ?= ./$(BACKBONE).pt
RESULT_FILE ?= $(METHOD)_$(SPLIT).json

define require_backbone_mode
	@if [ -z "$(BACKBONE)" ] || [ -z "$(MODE)" ]; then \
		echo "Usage: make $@ {isc|dino} {full|pred|gt}"; exit 1; \
	fi
endef

define require_method
	@if [ -z "$(METHOD)" ]; then \
		echo "Usage: make $@ {isc|dino} {full|pred|gt} {TN|SPD}"; exit 1; \
	fi
endef

.PHONY: extract simmap simmapviz vta eval

extract:
	$(call require_backbone_mode)
	python scripts/extract_features.py \
		--backbone $(BACKBONE) --crop_mode $(CROP_MODE) \
		$(YOLO_ARGS)

simmap:
	$(call require_backbone_mode)
	python scripts/vcsl_run_video_sim.py \
		--input-root $(ROOT)/features/$(BACKBONE)/$(MODE_PATH) \
		--output-root $(ROOT)/simmaps/$(BACKBONE)/$(MODE_PATH)

simmapviz:
	$(call require_backbone_mode)
	python scripts/visualize_similarity_maps.py \
		--input-root $(ROOT)/simmaps/$(BACKBONE)/$(MODE_PATH) \
		--output-root $(ROOT)/simmaps_png/$(BACKBONE)/$(MODE_PATH)

vta:
	$(call require_backbone_mode)
	$(call require_method)
	python scripts/vcsl_run_video_vta.py \
		--pair-file $(PAIR_FILE) \
		--input-root $(ROOT)/simmaps/$(BACKBONE)/$(MODE_PATH) \
		--output-root $(ROOT)/vta_out/$(BACKBONE)/$(MODE_PATH) \
		--alignment-method $(METHOD) --result-file $(RESULT_FILE) \
		$(if $(filter SPD,$(METHOD)),--spd-model-path $(SPD_MODEL),)

eval:
	$(call require_backbone_mode)
	$(call require_method)
	python scripts/vcsl_run_video_eval.py \
		--pred-file $(ROOT)/vta_out/$(BACKBONE)/$(MODE_PATH)/$(RESULT_FILE) \
		--split $(SPLIT)
