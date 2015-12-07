/* This file contains common definitions used in libmpti and in the ompt/pomp adapter for Score-P.
   It gets duplicated in these places and has to be kept in sync until libmpti gets integrated in Score-P! */

#ifndef MPTI_LIB_H
#define MPTI_LIB_H

// Use references
#define OMP_MPTI_REFERENCES 1

// MPTI version
#define MPTI_VERSION 100

// MPTI region types
#define MPTI_REGION_INVALID 0
#define MPTI_PARALLEL_REGION_ENTER 1
#define MPTI_PARALLEL_REGION_LEAVE 2
#define MPTI_TASK_ENTER 3
#define MPTI_TASK_LEAVE 4
#define MPTI_TARGET_REGION_ENTER 5
#define MPTI_TARGET_REGION_LEAVE 6

// MPTI region IDs
// End of custom region ids (start of uninstrumented region ids)
// Note: Pomp region IDs have a file prefix, so everything above 0x0000FFFF is from opari instrumented regions
#define MPTI_REGION_ID_CUSTOM_END (0xF00)
#define MPTI_REGION_ID_POMP_START (0x00010000)
#define MPTI_REGION_ID_BARRIER (MPTI_REGION_ID_CUSTOM_END)
#define MPTI_REGION_ID_IMPLICIT_TASK (MPTI_REGION_ID_CUSTOM_END - 1)

/* common OMPT command values */
#define OMPT_COMMAND_START 1
#define OMPT_COMMAND_PAUSE 2
#define OMPT_COMMAND_FLUSH 3
#define OMPT_COMMAND_STOP 4

/* own OMPT command values */
#define OMPT_COMMAND_GET_NUM_RECORDS 10
#define OMPT_COMMAND_FLUSH_RECORDS 11
#define OMPT_COMMAND_PUSH_REGION_ID 12
#define OMPT_COMMAND_POP_REGION_ID 13
#define OMPT_COMMAND_TARGET_BEGIN 14
#define OMPT_COMMAND_TARGET_END 15
#define OMPT_COMMAND_TASK_SET 16
#define OMPT_COMMAND_VERSION 100

#define INVALID_POMP_REGION_ID 0
#define INVALID_OMPT_PARALLEL_REGION_ID 0

/* measurement record */
typedef struct mpti_record_t
{
    uint64_t type;
    uint64_t time;
    uint64_t thread_id;
    uint64_t pomp_region_id;
#if ( OMP_MPTI_REFERENCES == 1 )
    uint64_t parallel_region_id;
    uint64_t parent_parallel_region_id;
#endif
} mpti_record_t;

#endif //MPTI_LIB_H
