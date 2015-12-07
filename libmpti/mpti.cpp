/*
 * Copyright (c) 2014
 * Technische Universitaet Dresden, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 */

/* MIC performance tools interface (MPTI) */

#include "mpti.hpp"

static int mpti_write_record(uint64_t regionType, uint64_t regionId, void* targetFunction, uint64_t parallelRegionId)
{
    //DEBUG_PRINTF("[MPTI] mpti_write_record\n");
    mpti_record_t record;
    int result    = OMPT_CALLBACK_SUCCESS;
    int thread_id = ompt_get_thread_id();

    record.thread_id                 = thread_id;
    record.time                      = omp_get_wtime() / omp_get_wtick();
    record.type                      = regionType;
#if ( OMP_MPTI_REFERENCES )
    record.parallel_region_id        = parallelRegionId;
    record.parent_parallel_region_id = ompt_get_parallel_id(1);
#endif

    /* for fixed POMP region IDs like e.g. barrier */
    if (regionId != INVALID_POMP_REGION_ID)
    {
        record.pomp_region_id = regionId;

	if (pomp_region_ids[thread_id].last_record)
	{
	    mpti_record_t *last_record = pomp_region_ids[thread_id].last_record;
	    if (regionId == MPTI_REGION_ID_BARRIER && regionType == MPTI_PARALLEL_REGION_ENTER &&
		last_record->type == MPTI_PARALLEL_REGION_LEAVE)
	    {
	        record.time = last_record->time;
	    }
	}
    }

    if (regionId == INVALID_POMP_REGION_ID)
    {
	/* need to find current regionID using per-thread stack */
        if (thread_id < pomp_region_ids.size())
        {
            // as tasks are executed asynchronous ...
            if ( regionType == MPTI_TASK_ENTER )
            {
                record.time = pomp_region_ids[thread_id].task_begin_time;
            }
            else if ( regionType == MPTI_TASK_LEAVE )
            {
                record.pomp_region_id = pomp_region_ids[thread_id].pomp_rid_task;
            }
            else if (!pomp_region_ids[thread_id].pomp_rid_stack.empty())
            {
                record.pomp_region_id = pomp_region_ids[thread_id].pomp_rid_stack.back().first;
                fprintf(stderr,"Stack not emptyMPTI Record POMP regin id: %d\n", record.pomp_region_id);
            } 
            else
            {
                // region has not been instrumented (e.g. function with declare target)
                if(targetFunction)
                {
                    #pragma omp critical
                    {           
                        map<void *, uint64_t>::const_iterator it = 
                                       outlined_functions.find(targetFunction);
                    
                        // if map already contains the outlined function 
                        if(outlined_functions.end() != it)
                        {
                            record.pomp_region_id = it->second;
                        } else
                        {   // insert the new outlined function with the corresponding region ID into the map
                            record.pomp_region_id = ++next_region_id;
                            outlined_functions.insert(
                              pair<void *,uint64_t>(targetFunction, record.pomp_region_id));
                        }
                        pomp_region_ids[thread_id].pomp_rid_stack.push_back(
                            make_pair(record.pomp_region_id, INVALID_OMPT_PARALLEL_REGION_ID));
                    }
                    fprintf(stderr,"MPTI Record POMP regin id: %d\n", record.pomp_region_id);
                }
                else
                {
                    fprintf(stderr, "No enter region for thread ID %d (record type: %d)\n", thread_id, regionType);
                    record.pomp_region_id = INVALID_POMP_REGION_ID;
                }
            }
        }
        else
        {
            fprintf(stderr, "Thread ID %d too high\n", thread_id);
            record.pomp_region_id = INVALID_POMP_REGION_ID;
        }
    }

    //DEBUG_PRINTF_ARGS("MPTI: write record - before critical - (%d,%d)\n", record.type, record.pomp_region_id);
#pragma omp critical
    if (mpti_state == MPTI_STATE_ON)
    {
        record_buffer[num_records] = record;
	pomp_region_ids[thread_id].last_record = &(record_buffer[num_records]);
        num_records++;

        if (num_records >= MAX_NUM_RECORDS)
        {
            num_records = -1;
            mpti_state = MPTI_STATE_ERROR;
            result = OMPT_CALLBACK_ERROR;
        }
    }
    //DEBUG_PRINTF_ARGS("MPTI: after critical %llu\n", regionId);

    return result;
}

static void mpti_flush_buffer(uint64_t** dst)
{
    DEBUG_PRINTF("[MPTI] mpti_flush_buffer\n");
            
    *dst = (uint64_t*) record_buffer;
    num_records = 0;
}

int mpti_callback_event_parallel_begin(
        ompt_task_id_t parent_task_id, /* id of parent task            */
        ompt_frame_t *parent_task_frame, /* frame data of parent task    */
        ompt_parallel_id_t parallel_id, /* id of parallel region        */
        uint32_t requested_team_size,     /* number of threads in team    */
        void *parallel_function /* pointer to outlined function */
        )
{
    if (mpti_state != MPTI_STATE_ON)
        return OMPT_CALLBACK_ERROR;

    DEBUG_PRINTF("[MPTI] mpti_callback_event_parallel_begin\n");

    return mpti_write_record(MPTI_PARALLEL_REGION_ENTER, INVALID_POMP_REGION_ID,
        parallel_function, parallel_id);
}

int mpti_callback_event_parallel_end(
        ompt_parallel_id_t parallel_id, /* id of parallel region        */
        ompt_task_id_t task_id /* id of task            */
        )
{
    if (mpti_state != MPTI_STATE_ON)
        return OMPT_CALLBACK_ERROR;

    DEBUG_PRINTF("[MPTI] mpti_callback_event_parallel_end\n");

    return mpti_write_record(MPTI_PARALLEL_REGION_LEAVE, INVALID_POMP_REGION_ID,
        NULL, parallel_id);
}

int mpti_callback_event_implicit_task_begin(
        ompt_parallel_id_t parallel_id, /* id of parallel region        */
        ompt_task_id_t task_id /* id of task            */
        )
{
    if (mpti_state != MPTI_STATE_ON)
        return OMPT_CALLBACK_ERROR;

    DEBUG_PRINTF("[MPTI] mpti_callback_event_implicit_task_begin");

    return mpti_write_record(MPTI_TASK_ENTER, MPTI_REGION_ID_IMPLICIT_TASK,
        NULL, parallel_id);
}

int mpti_callback_event_implicit_task_end(
        ompt_parallel_id_t parallel_id, /* id of parallel region        */
	ompt_task_id_t task_id /* id of task */
        )
{
    if (mpti_state != MPTI_STATE_ON)
        return OMPT_CALLBACK_ERROR;

    DEBUG_PRINTF("[MPTI] mpti_callback_event_implicit_task_end");

    return mpti_write_record(MPTI_TASK_LEAVE, MPTI_REGION_ID_IMPLICIT_TASK,
        NULL, parallel_id);
}

int mpti_callback_event_task_begin(
        ompt_task_id_t parent_task_id, /* ID of parent task */
        ompt_frame_t *parent_task_frame, /* frame data for parent task */
        ompt_task_id_t new_task_id, /* ID of created task */
        void *new_task_function /* pointer to outlined function */
        )
{
    if (mpti_state != MPTI_STATE_ON)
        return OMPT_CALLBACK_ERROR;

    DEBUG_PRINTF("[MPTI] mpti_callback_event_task_begin\n");

    //return mpti_write_record(MPTI_TASK_ENTER, INVALID_POMP_REGION_ID,
    //    new_task_function, INVALID_OMPT_PARALLEL_REGION_ID);
    
    // save time stamp in thread private data
    pomp_region_ids[ompt_get_thread_id()].task_begin_time = 
            omp_get_wtime() / omp_get_wtick();
}

int mpti_callback_event_task_end(
        ompt_task_id_t parent_task_id, /* ID of parent task */
        ompt_frame_t *parent_task_frame, /* frame data for parent task */
        ompt_task_id_t new_task_id, /* ID of created task */
        void *new_task_function /* pointer to outlined function */
        )
{
    if (mpti_state != MPTI_STATE_ON)
        return OMPT_CALLBACK_ERROR;

    DEBUG_PRINTF("[MPTI] mpti_callback_event_task_end\n");

    return mpti_write_record(MPTI_TASK_LEAVE, INVALID_POMP_REGION_ID,
        new_task_function, INVALID_OMPT_PARALLEL_REGION_ID);
}

void mpti_callback_event_barrier_begin(
        ompt_parallel_id_t parallel_id, /* ID of parallel region */
        ompt_task_id_t task_id /* ID of task */
        )
{
    if (mpti_state == MPTI_STATE_ON)
    {
        DEBUG_PRINTF("[MPTI] mpti_callback_event_barrier_begin\n");
        mpti_write_record(MPTI_PARALLEL_REGION_ENTER, MPTI_REGION_ID_BARRIER, NULL, parallel_id);
    }
}

void mpti_callback_event_barrier_end(
        ompt_parallel_id_t parallel_id, /* ID of parallel region */
        ompt_task_id_t task_id /* ID of task */
        )
{
    if (mpti_state == MPTI_STATE_ON)
    {
        DEBUG_PRINTF("[MPTI] mpti_callback_event_barrier_end\n");
        mpti_write_record(MPTI_PARALLEL_REGION_LEAVE, MPTI_REGION_ID_BARRIER, NULL, parallel_id);
    }
}

void mpti_callback_event_control(uint64_t command, uint64_t modifier)
{
    DEBUG_PRINTF_ARGS("[MPTI] control command %llu,%llu\n", command, modifier);
    switch (command)
    {
        case OMPT_COMMAND_START:
            if (record_buffer)
                mpti_state = MPTI_STATE_ON;
            else
                fprintf(stderr, "Warning: Can not enable measurement\n");
            break;

        case OMPT_COMMAND_PAUSE:
        case OMPT_COMMAND_STOP:
            mpti_state = MPTI_STATE_OFF;
            break;

        case OMPT_COMMAND_GET_NUM_RECORDS:
            *((int*) modifier) = num_records;
            break;

        case OMPT_COMMAND_FLUSH_RECORDS:
            mpti_flush_buffer((uint64_t**) modifier);
            break;

        case OMPT_COMMAND_PUSH_REGION_ID:
            unsigned int tid = (unsigned int) ompt_get_thread_id();
            if ( tid < MAX_NUM_THREADS )
            {
                pomp_region_ids[tid].pomp_rid_stack.push_back(
                    make_pair(modifier, INVALID_OMPT_PARALLEL_REGION_ID));
            }
            else
            {
                fprintf(stderr, "[MPTI] libmpti has been built for a maximum "
                        "of %d threads", MAX_NUM_THREADS);
            }
            break;

        case OMPT_COMMAND_POP_REGION_ID:
            unsigned int tid = (unsigned int) ompt_get_thread_id();
            if ( tid < MAX_NUM_THREADS )
            {
                pomp_region_ids[tid].pomp_rid_stack.pop_back();
            }
            else
            {
                fprintf(stderr, "[MPTI] libmpti has been built for a maximum "
                        "of %d threads", MAX_NUM_THREADS);
            }
            break;
            
        case OMPT_COMMAND_TARGET_BEGIN:
            mpti_write_record(MPTI_TARGET_REGION_ENTER, modifier, NULL,
                    INVALID_OMPT_PARALLEL_REGION_ID);
            break;
            
        case OMPT_COMMAND_TARGET_END:
            mpti_write_record(MPTI_TARGET_REGION_LEAVE, modifier, NULL,
                    INVALID_OMPT_PARALLEL_REGION_ID);
            break;
            
        // first command in a task
        case OMPT_COMMAND_TASK_SET:
            //set the pomp region ID in thread private data and 
            pomp_region_ids[ompt_get_thread_id()].pomp_rid_task = modifier;

            // write record
            mpti_write_record(MPTI_TASK_ENTER, modifier,
                NULL, INVALID_OMPT_PARALLEL_REGION_ID);
            break;

	case OMPT_COMMAND_VERSION:
	    *((uint64_t*) modifier) = MPTI_VERSION;
	    break;

        default:
            fprintf(stderr, "MPTI warning: Unhandled OMPT command (%llu)\n", command);
    }
    //DEBUG_PRINTF_ARGS("MPTI: control command %d done\n", command);
}

void mpti_callback_event_runtime_shutdown(void)
{
    DEBUG_PRINTF("[MPTI] mpti_callback_event_runtime_shutdown\n");
    if (record_buffer)
    {
        free(record_buffer);
        record_buffer = NULL;
    }

    mpti_state = MPTI_STATE_OFF;
}

extern "C" int ompt_initialize(
        ompt_function_lookup_t ompt_fn_lookup,
        const char *runtime_version,
        int ompt_version
        )
{
    int i;

    DEBUG_PRINTF_ARGS("[MPTI] ompt_initialize %s (ompt %d)\n", runtime_version, ompt_version);

    // lookup ompt_get_thread_id
    ompt_get_thread_id = (ompt_get_thread_id_t) ompt_fn_lookup("ompt_get_thread_id");
    // lookup ompt_get_task_id
    ompt_get_task_id = (ompt_get_task_id_t) ompt_fn_lookup("ompt_get_task_id");
    // lookup ompt_get_parallel_id
    ompt_get_parallel_id = (ompt_get_parallel_id_t) ompt_fn_lookup("ompt_get_parallel_id");
    // lookup ompt_set_callback
    int (*ompt_set_callback)(ompt_event_t, ompt_callback_t);
    ompt_set_callback = (ompt_set_callback_t) ompt_fn_lookup("ompt_set_callback");

    // register callbacks
    // mandatory
    MPTI_OMPT_SET_CALLBACK(event_parallel_begin);
    MPTI_OMPT_SET_CALLBACK(event_parallel_end);
    MPTI_OMPT_SET_CALLBACK(event_task_begin);
    MPTI_OMPT_SET_CALLBACK(event_task_end);
    MPTI_OMPT_SET_CALLBACK(event_control);
    MPTI_OMPT_SET_CALLBACK(event_runtime_shutdown);
    // optional
    MPTI_OMPT_SET_CALLBACK_OPTIONAL(event_barrier_begin);
    MPTI_OMPT_SET_CALLBACK_OPTIONAL(event_barrier_end);
#if 0
    MPTI_OMPT_SET_CALLBACK_OPTIONAL(event_implicit_task_begin);
    MPTI_OMPT_SET_CALLBACK_OPTIONAL(event_implicit_task_end);
#endif

    record_buffer = (mpti_record_t*) malloc(sizeof (mpti_record_t) * MAX_NUM_RECORDS);
    memset(record_buffer, 0, sizeof (mpti_record_t) * MAX_NUM_RECORDS);
    num_records = 0;

    /* Note that this is pretty unsafe.
     * It only works for devices with at most 512 threads and OMPT
     * does not guarantee to choose thread IDs of consecutive numbers
     * starting at 0. However, it works for now and is optimal in 
     * terms of concurrency */
    //mpti_thread_data thread_data[max_num_threads];
    mpti_thread_data *thread_data = 
            (mpti_thread_data *) malloc(sizeof (mpti_thread_data) * MAX_NUM_THREADS);
    for (i = 0; i < MAX_NUM_THREADS; ++i)
    {
        //pomp_region_ids.push_back(vector<pair<uint64_t, ompt_parallel_id_t> >());
        
        pomp_region_ids.push_back(thread_data[i]);
    }

    mpti_state = MPTI_STATE_OFF;

    return OMPT_INIT_SUCCESS;
}

/* Call from Fortran programs */
extern "C" void ompt_control_(int *command, int *modifier)
{
    //DEBUG_PRINTF_ARGS("ompt_control_(%d,%d)\n", (uint64_t)*command, (uint64_t)*modifier);
    mpti_callback_event_control((uint64_t)*command, (uint64_t)*modifier);
}
