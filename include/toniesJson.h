#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "error.h"

#define TEDDY_BENCH_AUDIO_ID_DEDUCT 0x50000000
#define SPECIAL_AUDIO_ID_ONE 1

typedef struct
{
    uint16_t no;
    char *model;
    uint32_t *audio_ids;
    uint8_t audio_ids_count;
    uint8_t *hashes;
    uint8_t hashes_count;
    char *title;
    char *series;
    char *episodes;
    char **tracks;
    uint8_t tracks_count;
    uint32_t release;
    char *language;
    char *category;
    char *picture;
} toniesJson_item_t;

void tonies_init();
error_t tonies_update();
error_t toniesV2_update();
void tonies_readJson(char *source, toniesJson_item_t **toniesCache, size_t *toniesCount);
toniesJson_item_t *tonies_byAudioId(uint32_t audio_id);
toniesJson_item_t *tonies_byAudioIdHash(uint32_t audio_id, uint8_t *hash);
toniesJson_item_t *tonies_byModel(char *model);
toniesJson_item_t *tonies_byAudioIdHashModel(uint32_t audio_id, uint8_t *hash, char *model);
bool tonies_byModelSeriesEpisode(char *model, char *series, char *episode, toniesJson_item_t *result[18], size_t *result_size);
void tonies_deinit();