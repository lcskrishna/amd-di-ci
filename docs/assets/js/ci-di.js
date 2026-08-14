// AMD distributed-inference (DI) grid view.
//
// Renders data/vllm/di/grid.json: the 20-cell model x shape x router matrix
// from buildkite.com/vllm/amd-distributed-inference-ci.
//
// Deliberately standalone rather than a branch inside ops-v2.js. ops-v2 owns
// its tabs via OWNED_TABS and returns early for anything else, so observing
// our own panel is sufficient — and it keeps this fork out of an 11k-line
// upstream file that would conflict on every rebase.
(function () {
  'use strict';

  var h = el;
  var TAB = 'ci-di';
  var SRC = 'data/vllm/di/grid.json';
  var STRIP = 12;

  // Mirrors TERMINAL_VERDICTS in build_di_grid.py. A queued or running step is
  // not evidence about a build and must not be counted as a failure.
  var TERMINAL = new Set(['passed', 'failed', 'soft_failed', 'timed_out', 'broken']);

  // The driver's failure_class, in escalating order of "this is a real bug".
  var CLASS_LABEL = {
    infra: 'Infra',
    bringup: 'Bringup',
    workload: 'Workload',
    ok: 'OK',
    unknown: 'Unclassified',
  };
  var CLASS_HELP = {
    infra: 'Never reached the workload — preflight rejection, allocation or node failure.',
    bringup: 'Cluster gave us nodes but the servers never came up.',
    workload: 'The test actually ran and returned a verdict. These are the real regressions.',
    unknown: 'No SLURM verdict line in the log (collected with --no-logs, or a non-terminal job).',
  };

  function verdictColor(v) {
    if (v === 'passed') return 'var(--accent-green)';
    if (v === 'failed' || v === 'broken' || v === 'timed_out') return 'var(--badge-closed)';
    if (v === 'soft_failed') return 'var(--accent-orange)';
    if (v === 'running' || v === 'waiting' || v === 'scheduled') return 'var(--accent-orange)';
    return 'var(--text-muted)';
  }

  function classColor(c) {
    if (c === 'workload') return 'var(--badge-closed)';
    if (c === 'bringup') return 'var(--accent-orange)';
    if (c === 'infra') return '#8957e5';
    if (c === 'ok') return 'var(--accent-green)';
    return 'var(--text-muted)';
  }

  // Tint a colour toward transparent. The colours above are CSS variables, so
  // the usual trick of appending an alpha suffix ("#ff000066") cannot work —
  // "var(--x)66" is not a colour and the declaration is dropped silently.
  function mix(c, percent) {
    return 'color-mix(in srgb, ' + c + ' ' + percent + '%, transparent)';
  }

  function mins(s) {
    if (s == null) return '--';
    return (s / 60).toFixed(1) + 'm';
  }

  function pct(r) {
    return Math.round(r * 100) + '%';
  }

  // ── Panels ───────────────────────────────────────────────────────────────

  function header(g) {
    var last = (g.builds && g.builds[0]) || null;
    var p = g.pipeline || {};
    var bits = [h('h2', {
      text: p.display_name || 'AMD Distributed Inference',
      style: { margin: '0 0 4px' },
    })];
    var sub = p.slug || 'amd-distributed-inference-ci';
    if (last) {
      sub += '  ·  latest build #' + last.build_number + ' (' + last.state + ')';
    }
    if (g.generated_at) sub += '  ·  collected ' + g.generated_at.slice(0, 16).replace('T', ' ');
    bits.push(h('div', { text: sub, style: { color: 'var(--text-muted)', fontSize: '13px' } }));
    return h('div', { style: { marginBottom: '18px' } }, bits);
  }

  function chip(value, label, c, tip) {
    return h('div', {
      title: tip || '',
      style: {
        border: '1px solid ' + mix(c, 40), background: mix(c, 10), borderRadius: '6px',
        padding: '10px 14px', minWidth: '96px',
      },
    }, [
      h('div', { text: String(value), style: { fontSize: '22px', fontWeight: '700', color: c } }),
      h('div', { text: label, style: { fontSize: '12px', color: 'var(--text-muted)' } }),
    ]);
  }

  // Passed leads, then the failure classes. Showing only failures made a run
  // of 18 infra rejections look identical whether the other two steps passed
  // or were still queued.
  function outcomeChips(passed, counts) {
    var chips = [chip(passed, 'Passed', 'var(--accent-green)',
      'Completed steps that passed.')];
    ['workload', 'bringup', 'infra', 'unknown'].forEach(function (k) {
      if (counts[k]) chips.push(chip(counts[k], CLASS_LABEL[k] || k, classColor(k), CLASS_HELP[k] || ''));
    });
    return chips;
  }

  function column(title, body, note) {
    return h('div', {
      style: {
        border: '1px solid var(--border)', borderRadius: '8px', padding: '12px 14px',
        minWidth: '0',
      },
    }, [
      h('div', {
        text: title,
        style: {
          fontSize: '12px', fontWeight: '700', color: 'var(--text-muted)',
          textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '10px',
        },
      }),
      body,
      note ? h('div', {
        text: note,
        style: { color: 'var(--text-muted)', fontSize: '11px', marginTop: '10px', lineHeight: '1.45' },
      }) : null,
    ].filter(Boolean));
  }

  // Counts for one build only. The aggregate answers "is this pipeline
  // healthy"; this answers "should I look at the run that just went red",
  // which is a different question and usually the urgent one.
  function lastBuildClasses(g) {
    var last = (g.build_rollup || [])[0];
    if (!last) return null;
    var counts = {};
    var passed = 0, completed = 0, pending = 0;
    Object.keys(last.models).forEach(function (m) {
      last.models[m].runs.forEach(function (r) {
        if (!TERMINAL.has(r.verdict)) { pending += 1; return; }
        completed += 1;
        if (r.verdict === 'passed') { passed += 1; return; }
        var k = r.failure_class || 'unknown';
        counts[k] = (counts[k] || 0) + 1;
      });
    });
    return {
      build_number: last.build_number, date: last.date,
      counts: counts, passed: passed, completed: completed, pending: pending,
    };
  }

  // Totals across every collected build. Derived from build_rollup so it
  // agrees with the table below by construction.
  function overallTotals(g) {
    var passed = 0, completed = 0;
    (g.build_rollup || []).forEach(function (b) {
      Object.keys(b.models).forEach(function (m) {
        passed += b.models[m].passed;
        completed += b.models[m].completed;
      });
    });
    return { passed: passed, completed: completed };
  }

  // The reason this dashboard exists: Buildkite says "14 failed", but most of
  // those never ran a test. Lead with that split — and with how many passed,
  // so a column of failure chips cannot be mistaken for the whole story.
  function failureClasses(g) {
    var fc = g.failure_classes || {};
    var totals = overallTotals(g);
    if (!Object.keys(fc).length && !totals.completed) return null;

    var onlyUnknown = Object.keys(fc).length === 1 && fc.unknown;
    var overallNote = onlyUnknown
      ? 'No SLURM verdicts yet — this pass ran with --no-logs. Re-run the collector with logs to split infra from real regressions.'
      : 'Every completed step across all collected builds, attributed from the SLURM driver’s own verdict line — which Buildkite’s red/green discards.';
    var overallCol = column(
      'Overall — ' + totals.passed + '/' + totals.completed + ' passed',
      h('div', { style: { display: 'flex', gap: '10px', flexWrap: 'wrap' } },
        outcomeChips(totals.passed, fc)),
      overallNote
    );

    var lastCol;
    var lb = lastBuildClasses(g);
    if (!lb) {
      lastCol = column('Last build', h('div', {
        text: 'No builds collected.',
        style: { color: 'var(--text-muted)', fontSize: '13px' },
      }));
    } else {
      var lastNote = 'Most recent build only'
        + (lb.pending ? ' — ' + lb.pending + ' step' + (lb.pending === 1 ? '' : 's')
            + ' still running, not counted either way.' : '.');
      lastCol = column(
        'Build #' + lb.build_number + ' — ' + lb.passed + '/' + lb.completed + ' passed',
        h('div', { style: { display: 'flex', gap: '10px', flexWrap: 'wrap' } },
          outcomeChips(lb.passed, lb.counts)),
        lastNote
      );
    }

    var hwRow = function (label, value) {
      return h('div', {
        style: { display: 'flex', gap: '10px', fontSize: '13px', padding: '2px 0' },
      }, [
        h('span', { text: label, style: { color: 'var(--text-muted)', minWidth: '74px' } }),
        h('span', { text: value, style: { fontWeight: '600' } }),
      ]);
    };

    var hwCol = column('Hardware', h('div', {}, [
      hwRow('Accelerator', 'AMD MI355X'),
      hwRow('Fabric', 'AINIC'),
      hwRow('Queue', 'amd_mi350_ainic'),
      hwRow('Topology', '1P1D = 2 nodes · 2P2D = 4 nodes'),
      hwRow('Parallelism', (g.axes || {}).mode || 'TP8'),
      hwRow('KV transport', (g.axes || {}).transport || 'MoRIIO'),
    ]), 'Fixed for every cell in the grid; the only hardware axis is node count.');

    return h('div', { style: { marginBottom: '22px' } }, [
      h('h3', { text: 'Summary', style: { margin: '0 0 10px', fontSize: '15px' } }),
      h('div', {
        style: {
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: '12px', alignItems: 'start',
        },
      }, [
        overallCol,
        lastCol,
        hwCol,
      ]),
    ]);
  }

  // ── Build x model ────────────────────────────────────────────────────────

  // Read the rollup the collector emits rather than folding up cell.history:
  // that field is capped at the flaky window, so the middle builds are absent
  // from it and a table built that way would show holes.
  function pivot(g) {
    return g.build_rollup || [];
  }

  function ratioColor(p, t) {
    if (!t) return 'var(--text-muted)';
    var r = p / t;
    if (r === 1) return 'var(--accent-green)';
    if (r === 0) return 'var(--badge-closed)';
    return 'var(--accent-orange)';
  }

  function detailRow(row, models, span) {
    var blocks = models.filter(function (m) { return row.models[m]; }).map(function (m) {
      var d = row.models[m];
      var runs = d.runs.slice().sort(function (a, b) {
        return (a.shape + a.router).localeCompare(b.shape + b.router);
      }).map(function (r) {
        var tag = r.failure_class && r.failure_class !== 'ok' ? r.failure_class : '';
        return h('div', {
          style: { display: 'flex', gap: '10px', alignItems: 'baseline', fontSize: '12px', padding: '2px 0' },
        }, [
          h('span', {
            text: r.shape + ' ' + r.router,
            style: { minWidth: '150px', color: 'var(--text-muted)' },
          }),
          h('a', {
            text: r.verdict, href: r.job_url || '#', target: '_blank', rel: 'noopener',
            style: { minWidth: '70px', color: verdictColor(r.verdict), textDecoration: 'none', fontWeight: '600' },
          }),
          h('span', { text: mins(r.runtime_s), style: { minWidth: '55px', color: 'var(--text-muted)' } }),
          tag ? h('span', {
            text: tag + (r.slurm_state ? ' · ' + r.slurm_state : ''),
            title: r.reason || '',
            style: {
              fontSize: '11px', color: classColor(tag),
              border: '1px solid ' + mix(classColor(tag), 40), borderRadius: '3px', padding: '0 5px',
            },
          }) : null,
        ].filter(Boolean));
      });
      return h('div', { style: { marginBottom: '10px', minWidth: '330px' } }, [
        h('div', {
          text: m + '  ' + d.passed + '/' + d.completed,
          style: { fontSize: '12px', fontWeight: '700', marginBottom: '2px' },
        }),
      ].concat(runs));
    });

    return h('tr', { style: { display: 'none' }, 'data-detail': String(row.build_number) }, [
      h('td', { colspan: String(span), style: { padding: '10px 14px', background: 'var(--card-bg)' } }, [
        h('div', { style: { display: 'flex', gap: '26px', flexWrap: 'wrap' } }, blocks),
      ]),
    ]);
  }

  function buildModelTable(g) {
    var models = (g.axes || {}).models || [];
    var rows = pivot(g);
    if (!rows.length) return null;

    var th = function (t, alignRight) {
      return h('th', {
        text: t,
        style: {
          textAlign: alignRight ? 'right' : 'left', padding: '6px 8px', fontSize: '12px',
          color: 'var(--text-muted)', borderBottom: '1px solid var(--border)',
          fontWeight: '600', whiteSpace: 'nowrap',
        },
      });
    };

    var head = h('tr', {}, [th('Build'), th('Date')]
      .concat(models.map(function (m) { return th(m, true); }))
      .concat([th('Total', true)]));

    var body = [];
    rows.forEach(function (r) {
      var tp = 0, tt = 0;
      var cells = models.map(function (m) {
        var d = r.models[m];
        if (!d || !d.completed) {
          return h('td', { text: '--', style: { padding: '6px 8px', textAlign: 'right', color: 'var(--text-muted)', fontSize: '13px' } });
        }
        tp += d.passed; tt += d.completed;
        return h('td', {
          text: d.passed + '/' + d.completed,
          style: {
            padding: '6px 8px', textAlign: 'right', fontSize: '13px',
            fontWeight: '600', color: ratioColor(d.passed, d.completed),
          },
        });
      });

      var tr = h('tr', {
        'data-build': String(r.build_number),
        style: { cursor: 'pointer', borderTop: '1px solid var(--border)' },
      }, [
        h('td', { style: { padding: '6px 8px', fontSize: '13px', whiteSpace: 'nowrap' } }, [
          h('span', { text: '▸ ', style: { color: 'var(--text-muted)' } }),
          h('span', { text: '#' + r.build_number, style: { fontWeight: '600' } }),
        ]),
        h('td', { text: (r.date || '').slice(5), style: { padding: '6px 8px', fontSize: '13px', color: 'var(--text-muted)' } }),
      ].concat(cells).concat([
        h('td', {
          text: tp + '/' + tt,
          style: { padding: '6px 8px', textAlign: 'right', fontSize: '13px', fontWeight: '700', color: ratioColor(tp, tt) },
        }),
      ]));

      var detail = detailRow(r, models, models.length + 3);
      tr.addEventListener('click', function () {
        var open = detail.style.display !== 'none';
        detail.style.display = open ? 'none' : 'table-row';
        var caret = tr.querySelector('span');
        if (caret) caret.textContent = open ? '▸ ' : '▾ ';
      });
      body.push(tr, detail);
    });

    return h('div', { style: { marginBottom: '22px', overflowX: 'auto' } }, [
      h('h3', { text: 'Builds × models', style: { margin: '0 0 8px', fontSize: '15px' } }),
      h('table', { style: { width: '100%', borderCollapse: 'collapse', fontSize: '14px' } }, [
        h('thead', {}, [head]),
        h('tbody', {}, body),
      ]),
      h('div', {
        text: 'passed/completed across each model’s four cells (1P1D+2P2D × proxy+vllm-router). '
          + 'Click a build to expand every step, its runtime, and the SLURM failure class. '
          + 'Denominators come from steps actually run, so early partial builds are not scored out of 20.',
        style: { fontSize: '12px', color: 'var(--text-muted)', marginTop: '8px' },
      }),
    ]);
  }

  function strip(cell) {
    // history is newest-first; render oldest -> newest so time reads left to right.
    var hist = (cell.history || []).slice(0, STRIP).reverse();
    var boxes = hist.map(function (e) {
      var tip = '#' + e.build_number + '  ' + e.verdict
        + (e.failure_class ? '  [' + e.failure_class + ']' : '')
        + (e.slurm_state ? '  ' + e.slurm_state : '')
        + '\n' + (e.date || '') + '  ' + mins(e.runtime_s);
      var box = h('a', {
        href: e.job_url || '#',
        target: '_blank',
        rel: 'noopener',
        title: tip,
        style: {
          display: 'block', width: '9px', height: '16px', borderRadius: '2px',
          // Everything stays in the verdict's own hue — no neutral dark edge.
          // A genuine test failure (the rare, important case) is picked out by
          // a denser fill rather than by an outline in the text colour.
          background: mix(verdictColor(e.verdict), e.failure_class === 'workload' ? 62 : 26),
          border: '1px solid ' + mix(verdictColor(e.verdict), 38),
          boxSizing: 'border-box',
          textDecoration: 'none',
        },
      });
      return box;
    });
    return h('div', { style: { display: 'flex', gap: '2px', marginTop: '5px' } }, boxes);
  }

  function cellBox(cell) {
    if (!cell) {
      return h('td', { text: '--', style: { padding: '8px', color: 'var(--text-muted)' } });
    }
    if (!cell.enabled) {
      return h('td', { style: { padding: '8px', verticalAlign: 'top' } }, [
        h('div', {
          text: 'disabled',
          title: 'Commented out in pipeline-disagg.yaml — the wide-EP block.',
          style: {
            fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic',
            border: '1px dashed var(--border)', borderRadius: '5px', padding: '10px', textAlign: 'center',
          },
        }),
      ]);
    }

    var rateTxt, rateColor;
    if (cell.pass_rate == null) {
      rateTxt = 'never run';
      rateColor = 'var(--text-muted)';
    } else if (!cell.rate_is_reportable) {
      // A confident percentage from three 120-minute samples is a lie.
      rateTxt = cell.passed + '/' + cell.completed + '  (n too low)';
      rateColor = 'var(--accent-orange)';
    } else {
      rateTxt = pct(cell.pass_rate) + '  (' + cell.passed + '/' + cell.completed + ')';
      rateColor = cell.pass_rate >= 0.9 ? 'var(--accent-green)'
        : cell.pass_rate >= 0.5 ? 'var(--accent-orange)' : 'var(--badge-closed)';
    }

    var meta = [];
    if (cell.median_runtime_s != null) meta.push('median ' + mins(cell.median_runtime_s));
    if (cell.flips) meta.push(cell.flips + ' flips');

    return h('td', {
      style: {
        padding: '8px', verticalAlign: 'top',
        borderLeft: '3px solid ' + mix(verdictColor(cell.last_verdict), 55),
      },
    }, [
      h('div', { text: rateTxt, style: { fontSize: '13px', fontWeight: '600', color: rateColor } }),
      strip(cell),
      h('div', {
        text: meta.join('  ·  '),
        style: { fontSize: '11px', color: 'var(--text-muted)', marginTop: '5px' },
      }),
    ]);
  }

  function gridTable(g) {
    var axes = g.axes || {};
    var models = axes.models || [];
    var shapes = axes.shapes || [];
    var routers = axes.routers || [];
    var mode = axes.mode || 'TP8';

    // Index the enumerated cells so a missing combination renders blank
    // rather than shifting every column after it.
    //
    // mode MUST be part of the key. The wide-EP cell is
    // DeepSeek-V3|1P1D|EP8/DP8-WideEP|MoRIIO|proxy — identical to a live cell
    // in model, shape and router, differing only in mode. Keying without it
    // lets the disabled cell overwrite the live one.
    var byKey = {};
    (g.cells || []).forEach(function (c) {
      byKey[[c.model, c.shape, c.mode, c.router].join('|')] = c;
    });

    var cols = [];
    shapes.forEach(function (s) {
      routers.forEach(function (r) { cols.push({ shape: s, router: r }); });
    });

    var th = function (t, extra) {
      var st = {
        textAlign: 'left', padding: '6px 8px', fontSize: '12px',
        color: 'var(--text-muted)', borderBottom: '1px solid var(--border)',
        fontWeight: '600', whiteSpace: 'nowrap',
      };
      Object.assign(st, extra || {});
      return h('th', { text: t, style: st });
    };

    var head = h('tr', {}, [th('Model')].concat(cols.map(function (c) {
      return th(c.shape + ' · ' + c.router);
    })));

    var rows = models.map(function (m) {
      var cells = cols.map(function (c) {
        return cellBox(byKey[[m, c.shape, mode, c.router].join('|')]);
      });
      var name = h('td', {
        style: {
          padding: '8px', fontSize: '13px', fontWeight: '600',
          whiteSpace: 'nowrap', borderTop: '1px solid var(--border)',
        },
      }, [h('span', { text: m })]);
      return h('tr', {}, [name].concat(cells));
    });

    // Anything the enumeration did not predict: a renamed or retired step.
    var extras = (g.cells || []).filter(function (c) { return c.unexpected; });
    extras.forEach(function (c) {
      var name = h('td', {
        style: {
          padding: '8px', fontSize: '13px', whiteSpace: 'nowrap',
          borderTop: '1px solid var(--border)', color: 'var(--accent-orange)',
        },
      }, [h('span', { text: c.model + ' (unexpected)', title: c.cell_id })]);
      var pad = cols.map(function (col, i) {
        return i === 0 ? cellBox(c) : h('td', { text: '' });
      });
      rows.push(h('tr', {}, [name].concat(pad)));
    });

    return h('div', { style: { marginBottom: '22px', overflowX: 'auto' } }, [
      h('h3', { text: 'Grid', style: { margin: '0 0 8px', fontSize: '15px' } }),
      h('table', { style: { width: '100%', borderCollapse: 'collapse', fontSize: '14px' } }, [
        h('thead', {}, [head]),
        h('tbody', {}, rows),
      ]),
      h('div', {
        text: 'Each square is one build, oldest left. Outlined squares are genuine workload '
          + 'failures; click any square to open the Buildkite job. Rates are withheld below '
          + (g.min_samples_for_rate || 5) + ' completed runs.',
        style: { fontSize: '12px', color: 'var(--text-muted)', marginTop: '8px' },
      }),
    ]);
  }

  // Modes outside the main matrix — today just the commented-out wide-EP
  // block. Rendered explicitly so its eventual enablement is visible rather
  // than silent.
  function offMatrix(g) {
    var mode = (g.axes || {}).mode || 'TP8';
    var others = (g.cells || []).filter(function (c) {
      return c.mode && c.mode !== mode && !c.unexpected;
    });
    if (!others.length) return null;

    var items = others.map(function (c) {
      var live = c.enabled && c.attempts > 0;
      var status = !c.enabled ? 'disabled in pipeline-disagg.yaml'
        : (c.attempts ? c.completed + ' runs' : 'enabled, not yet run');
      return h('div', {
        style: {
          border: '1px dashed var(--border)', borderRadius: '6px',
          padding: '10px 14px', minWidth: '260px',
        },
      }, [
        h('div', {
          text: c.model + ' · ' + c.shape + ' · ' + c.router,
          style: { fontSize: '13px', fontWeight: '600' },
        }),
        h('div', {
          text: c.mode,
          style: { fontSize: '12px', color: 'var(--accent)', fontFamily: 'monospace' },
        }),
        h('div', {
          text: status,
          style: { fontSize: '12px', color: live ? 'var(--text)' : 'var(--text-muted)', marginTop: '4px' },
        }),
        live ? strip(c) : null,
      ].filter(Boolean));
    });

    return h('div', { style: { marginBottom: '22px' } }, [
      h('h3', { text: 'Wide-EP watch', style: { margin: '0 0 8px', fontSize: '15px' } }),
      h('div', { style: { display: 'flex', gap: '10px', flexWrap: 'wrap' } }, items),
      h('div', {
        text: 'Kept out of the matrix above because it shares model, shape and router '
          + 'with a live cell and differs only by mode.',
        style: { fontSize: '12px', color: 'var(--text-muted)', marginTop: '8px' },
      }),
    ]);
  }

  function unclassified(g) {
    var u = g.unclassified || [];
    if (!u.length) return null;
    // Label format is the schema: a renamed step silently moves its cell, so
    // this bucket must be loud rather than swallowed.
    var seen = {};
    u.forEach(function (r) {
      var k = r.label || '(empty)';
      if (!seen[k]) seen[k] = { label: k, n: 0, last: r.build_number };
      seen[k].n += 1;
      if (r.build_number > seen[k].last) seen[k].last = r.build_number;
    });
    var items = Object.keys(seen).map(function (k) {
      var s = seen[k];
      return h('li', {
        text: s.label + '  —  ' + s.n + ' run' + (s.n === 1 ? '' : 's') + ', last in #' + s.last,
        style: { fontSize: '13px', marginBottom: '3px' },
      });
    });
    return h('div', {
      style: {
        border: '1px solid var(--accent-orange)66', background: 'var(--accent-orange)14',
        borderRadius: '6px', padding: '12px 14px', marginBottom: '22px',
      },
    }, [
      h('div', {
        text: 'Unrecognised step labels (' + items.length + ')',
        style: { fontWeight: '600', fontSize: '14px', marginBottom: '6px' },
      }),
      h('ul', { style: { margin: '0', paddingLeft: '18px', color: 'var(--text-muted)' } }, items),
    ]);
  }

  function agentsPanel(g) {
    var a = (g.agents || []).filter(function (x) { return x.completed > 0; });
    if (!a.length) return null;
    var rows = a.slice(0, 10).map(function (x) {
      return h('tr', {}, [
        h('td', { text: x.agent_name, style: { padding: '5px 8px', fontSize: '13px' } }),
        h('td', { text: String(x.completed), style: { padding: '5px 8px', fontSize: '13px' } }),
        h('td', { text: String(x.failed), style: { padding: '5px 8px', fontSize: '13px' } }),
        h('td', {
          text: pct(x.failure_rate),
          style: {
            padding: '5px 8px', fontSize: '13px', fontWeight: '600',
            color: x.failure_rate >= 0.5 ? 'var(--badge-closed)' : 'var(--text)',
          },
        }),
      ]);
    });
    var th = function (t) {
      return h('th', {
        text: t,
        style: {
          textAlign: 'left', padding: '5px 8px', fontSize: '12px',
          color: 'var(--text-muted)', borderBottom: '1px solid var(--border)',
        },
      });
    };
    return h('details', { style: { marginBottom: '14px' } }, [
      h('summary', {
        text: 'Agent attribution (' + a.length + ')',
        style: { cursor: 'pointer', fontSize: '14px', fontWeight: '600', marginBottom: '8px' },
      }),
      h('table', { style: { width: '100%', borderCollapse: 'collapse', marginTop: '8px' } }, [
        h('thead', {}, [h('tr', {}, [th('Agent'), th('Completed'), th('Failed'), th('Failure rate')])]),
        h('tbody', {}, rows),
      ]),
      h('div', {
        text: 'These are SLURM login nodes, not the compute nodes that ran the test — '
          + 'the Buildkite agent only submits the job. Treat a hot row as a submission-path '
          + 'problem, not a bad GPU node.',
        style: { fontSize: '12px', color: 'var(--text-muted)', marginTop: '6px' },
      }),
    ]);
  }

  function buildsPanel(g) {
    var b = g.builds || [];
    if (!b.length) return null;
    var rows = b.slice(0, 25).map(function (x) {
      return h('tr', {}, [
        h('td', { style: { padding: '5px 8px', fontSize: '13px' } }, [
          h('a', {
            text: '#' + x.build_number, href: x.build_url, target: '_blank', rel: 'noopener',
            style: { color: 'var(--accent)', textDecoration: 'none' },
          }),
        ]),
        h('td', {
          text: x.state,
          style: { padding: '5px 8px', fontSize: '13px', color: verdictColor(x.state) },
        }),
        h('td', { text: (x.created_at || '').slice(0, 16).replace('T', ' '), style: { padding: '5px 8px', fontSize: '13px' } }),
        h('td', { text: x.branch || '', style: { padding: '5px 8px', fontSize: '12px', color: 'var(--text-muted)' } }),
        h('td', { text: (x.commit || '').slice(0, 8), style: { padding: '5px 8px', fontSize: '12px', fontFamily: 'monospace', color: 'var(--text-muted)' } }),
      ]);
    });
    var th = function (t) {
      return h('th', {
        text: t,
        style: {
          textAlign: 'left', padding: '5px 8px', fontSize: '12px',
          color: 'var(--text-muted)', borderBottom: '1px solid var(--border)',
        },
      });
    };
    return h('details', {}, [
      h('summary', {
        text: 'Builds (' + b.length + ')',
        style: { cursor: 'pointer', fontSize: '14px', fontWeight: '600' },
      }),
      h('table', { style: { width: '100%', borderCollapse: 'collapse', marginTop: '8px' } }, [
        h('thead', {}, [h('tr', {}, [th('Build'), th('State'), th('Created'), th('Branch'), th('Commit')])]),
        h('tbody', {}, rows),
      ]),
    ]);
  }

  // ── Render ───────────────────────────────────────────────────────────────

  async function render() {
    var host = document.getElementById(TAB + '-view');
    if (!host) return;
    host.innerHTML = '';
    host.append(h('div', { text: 'Loading DI grid...', style: { color: 'var(--text-muted)' } }));

    var g = await fetchJSON(SRC, { timeoutMs: 10000 });
    host.innerHTML = '';

    if (!g) {
      host.append(h('div', { style: { color: 'var(--text-muted)' } }, [
        h('p', { text: 'No DI data published yet.' }),
        h('p', {
          text: 'Run: python scripts/collect_di_ci.py --days 30 --output data/vllm/di/',
          style: { fontFamily: 'monospace', fontSize: '12px' },
        }),
      ]));
      return;
    }

    var parts = [header(g), failureClasses(g), gridTable(g), offMatrix(g),
                 buildModelTable(g), unclassified(g), agentsPanel(g), buildsPanel(g)];
    parts.forEach(function (p) { if (p) host.append(p); });
  }

  // ops-v2 owns its own tabs and ignores this one, so the panel gaining
  // .active is our only signal.
  function hook() {
    var panel = document.getElementById('tab-' + TAB);
    if (!panel) return;
    var go = function () {
      if (panel.classList.contains('active') && !panel.dataset.loaded) {
        panel.dataset.loaded = '1';
        render();
      }
    };
    new MutationObserver(go).observe(panel, { attributes: true, attributeFilter: ['class'] });
    go();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', hook);
  } else {
    hook();
  }

  window.__ciDi = { render: render };
})();
