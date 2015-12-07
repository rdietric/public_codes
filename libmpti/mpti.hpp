/*
 * Copyright (c) 2014
 * Technische Universitaet Dresden, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 */

/* MIC performance tools interface (MPTI) */

#ifndef MPTI_H
#define MPTI_H

#include <omp.h>
#include <ompt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>

#include "pomp2_dummy.h"
#include "mpti_lib.h"

using namespace std;

// MPTI library debugging
#if 0
#define DEBUG_PRINTF(format) fprintf (stderr, format)
#define DEBUG_PRINTF_ARGS(format, ...) fprintf (stderr, format, __VA_ARGS__); fflush(stderr)
#else
#define DEBUG_PRINTF(format)
#define DEBUG_PRINTF_ARGS(format, ...)
#endif

#define MAX_NUM_RECORDS (1024*1024*64)

#define MAX_NUM_THREADS 512

/* internal measurement library state options */
typedef enum mpti_state_t
{
    MPTI_STATE_OFF = 0,
    MPTI_STATE_ON = 1,
    MPTI_STATE_ERROR = 2
} mpti_state_t;

/* OMPT callback return values */
#define OMPT_CALLBACK_SUCCESS 3
#define OMPT_CALLBACK_ERROR 0

/* OMPT init return values */
#define OMPT_INIT_SUCCESS 1

/* macro to register a callback and print error on failure */
#define MPTI_SET_CALLBACK(_cb_name)                                                              \
    ompt_set_callback(ompt_##_cb_name, (ompt_callback_t) mpti_callback_##_cb_name)

#define MPTI_OMPT_SET_CALLBACK(_cb_name)                                                         \
{                                                                                                \
       int _status = MPTI_SET_CALLBACK(_cb_name);						 \
       if (_status != ompt_set_result_event_may_occur_callback_always)                           \
       { fprintf(stderr, "Error: Failed to set callback for ompt_%s (status %d)\n", #_cb_name, _status); }            \
}

#define MPTI_OMPT_SET_CALLBACK_OPTIONAL(_cb_name)                                                \
{                                                                                                \
       int _status = MPTI_SET_CALLBACK(_cb_name);						 \
       if (_status != ompt_set_result_event_may_occur_callback_always)                           \
       { fprintf(stderr, "Warning: Failed to set optional callback for ompt_%s (status %d)\n", #_cb_name, _status); } \
}

/* internal measurement library state */
static mpti_record_t *record_buffer = NULL;
static int num_records = 0;
static uint64_t next_region_id = MPTI_REGION_ID_CUSTOM_END + 1;
static mpti_state_t mpti_state = MPTI_STATE_OFF;

typedef struct mpti_thread_data
{
  /* pair is <POMP id, OMPT parallel id> */
  vector< pair<uint64_t, ompt_parallel_id_t> > pomp_rid_stack;
  uint64_t pomp_rid_task;
  uint64_t task_begin_time;
  mpti_record_t* last_record;

  mpti_thread_data()
  {
    pomp_rid_stack = vector<pair<uint64_t, ompt_parallel_id_t> >();
    pomp_rid_task = INVALID_POMP_REGION_ID;
    task_begin_time = 0;
    last_record = NULL;
  }
}mpti_thread_data;
static vector<mpti_thread_data> pomp_region_ids;

/* pair is <outlined function pointer, MPTI region ID> */
static map<void *, uint64_t> outlined_functions;

/* inquiry functions */
static ompt_thread_id_t(*ompt_get_thread_id)(void) = NULL;
static ompt_task_id_t(*ompt_get_task_id)(int depth) = NULL;
static ompt_parallel_id_t(*ompt_get_parallel_id)(int ancestor_level) = NULL;

#endif
