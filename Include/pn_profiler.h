#ifndef PN_PROFILER_H
#define PN_PROFILER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>

#include <pn_riscv_defs.h>

#define PN_PROFILER_MAX_EVENTS 256u
#define PN_PROFILER_INVALID_HANDLE UINT32_MAX

typedef struct
{
    const char *tag;
    uint64_t start_cycle;
    uint64_t end_cycle;
    uint64_t start_instret;
    uint64_t end_instret;
    uint64_t start_data;
    uint64_t end_data;
    bool completed;
} pn_profiler_event;

typedef struct
{
    pn_profiler_event events[PN_PROFILER_MAX_EVENTS];
    size_t num_events;
} pn_profiler;

/* RV32: retry if the low word rolls over during the read. */
static inline uint64_t pn_profiler_read_cycle(void)
{
    uint32_t lo, hi1, hi2;
    do
    {
        hi1 = read_csr(cycleh);
        lo = read_csr(cycle);
        hi2 = read_csr(cycleh);
    } while (hi1 != hi2);

    return ((uint64_t)hi1 << 32) | lo;
}

static inline uint64_t pn_profiler_read_instret(void)
{
    uint32_t lo, hi1, hi2;
    do
    {
        hi1 = read_csr(instreth);
        lo = read_csr(instret);
        hi2 = read_csr(instreth);
    } while (hi1 != hi2);

    return ((uint64_t)hi1 << 32) | lo;
}

/* Expand numeric CSR macros before read_csr stringifies them. */
#define PN_PROFILER_READ_CSR(reg) read_csr(reg)

static inline uint64_t pn_profiler_read_data(void)
{
    uint32_t lo, hi1, hi2;
    do
    {
        hi1 = PN_PROFILER_READ_CSR(CSR_PN_DATACOUNTH);
        lo = PN_PROFILER_READ_CSR(CSR_PN_DATACOUNT);
        hi2 = PN_PROFILER_READ_CSR(CSR_PN_DATACOUNTH);
    } while (hi1 != hi2);

    return ((uint64_t)hi1 << 32) | lo;
}

#undef PN_PROFILER_READ_CSR

/* Also resets an existing profiler, discarding its recorded events. */
static inline void pn_profiler_init(pn_profiler *profiler)
{
    if (profiler != NULL)
    {
        profiler->num_events = 0;
    }
}

static inline uint32_t pn_profiler_begin_event(pn_profiler *profiler, const char *tag)
{
    if (profiler == NULL)
    {
        return PN_PROFILER_INVALID_HANDLE;
    }

    if (profiler->num_events >= PN_PROFILER_MAX_EVENTS)
    {
        profiler->num_events = 0;
    }

    uint32_t handle = (uint32_t)profiler->num_events;
    pn_profiler_event *event = &profiler->events[handle];

    event->tag = tag;
    event->completed = false;
    event->start_cycle = pn_profiler_read_cycle();
    event->start_instret = pn_profiler_read_instret();
    event->start_data = pn_profiler_read_data();

    ++profiler->num_events;
    return handle;
}

static inline void pn_profiler_end_event(pn_profiler *profiler, uint32_t handle)
{
    if (profiler == NULL || handle >= profiler->num_events)
    {
        return;
    }

    pn_profiler_event *event = &profiler->events[handle];
    event->end_cycle = pn_profiler_read_cycle();
    event->end_instret = pn_profiler_read_instret();
    event->end_data = pn_profiler_read_data();
    event->completed = true;
}

static inline void pn_profiler_report(const pn_profiler *profiler)
{
    if (profiler == NULL)
    {
        return;
    }

    printf("     %-20s %10s %12s %10s\n", "Tag", "Cycles", "Instructions", "Data");

    for (size_t i = 0; i < profiler->num_events; ++i)
    {
        const pn_profiler_event *event = &profiler->events[i];

        if (!event->completed)
        {
            continue;
        }

        printf("%3zu: %-20s %10" PRIu64 " %12" PRIu64 " %10" PRIu64 "\n",
               i,
               event->tag != NULL ? event->tag : "(null)",
               event->end_cycle - event->start_cycle,
               event->end_instret - event->start_instret,
               event->end_data - event->start_data);
    }
}

#endif /* PN_PROFILER_H */
