#!/usr/bin/env python3
"""Make OpenMP worksharing loops honour OMP_SCHEDULE by giving them schedule(runtime).

ORBIT applies the swept schedule and chunk with omp_set_schedule(), which only affects loops
written with schedule(runtime). In C/C++ and free-form Fortran sources this script

  * adds schedule(runtime) to worksharing loops without a schedule clause, and
  * replaces schedule(static|dynamic|guided|auto[, n]) by schedule(runtime).

Only worksharing-loop constructs are touched: "for", "parallel for", "for simd",
"parallel for simd" (Fortran: the "do" equivalents). Offload constructs (target, teams,
distribute), simd, taskloop and sections are left alone.

nowait loops need care: with a static schedule, iteration i runs on the same thread in
consecutive loops, so a loop may read what a preceding nowait loop wrote without a barrier.
With dynamic scheduling that becomes a data race. Therefore, unless the loop is already
dynamic/guided/auto (it cannot rely on a mapping then), the script skips

  * every loop with nowait, and
  * the loops that follow a nowait loop up to the next barrier or parallel region,

and lists them. Loops in files named with --allow-nowait are patched anyway; use it only
for loops that were reviewed and do not depend on the iteration-to-thread mapping.

The script is idempotent (loops that already use schedule(runtime) are left as they are) and
works with Python 3.6+.

Usage:
    patch_schedule_runtime.py ROOT [ROOT ...] [--dry-run] [--backup] [--report FILE]
                              [--allow-nowait PATH_SUBSTRING ...] [--exclude PATH_SUBSTRING ...]
"""
import argparse
import os
import re
import shutil
import sys

C_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
F_SUFFIXES = {".f90", ".f95", ".f03"}  # free-form only
SKIP_DIRS = {".git"}
ADAPTIVE = {"dynamic", "guided", "auto", "runtime"}  # no static iteration->thread mapping to rely on
MAX_FORTRAN_LINE = 132

C_LOOP_WORDS = {"parallel", "for", "simd"}
F_LOOP_WORDS = {"parallel", "do", "simd"}
C_LOOPS = {("for",), ("for", "simd"), ("parallel", "for"), ("parallel", "for", "simd")}
F_LOOPS = {("do",), ("do", "simd"), ("parallel", "do"), ("parallel", "do", "simd")}

C_START = re.compile(r"^[ \t]*#[ \t]*pragma[ \t]+omp\b")
F_START = re.compile(r"^[ \t]*!\$omp\b", re.I)
WORD_C = re.compile(r"(?:[ \t]|\\\n)*([A-Za-z_]\w*)")
WORD_F = re.compile(r"[ \t]*([A-Za-z_]\w*)")
SCHEDULE = re.compile(r"\bschedule\s*\(", re.I)
SCHEDULE_MODIFIERS = re.compile(
    r"^\s*(?:monotonic|nonmonotonic|simd)\s*(?:,\s*(?:monotonic|nonmonotonic|simd)\s*)*:", re.I)
NOWAIT = re.compile(r"\bnowait\b", re.I)


def blank(match):
    return re.sub(r"[^\n]", " ", match.group(0))


def mask_comments(text, fortran):
    """Blank out comments, keeping every offset unchanged."""
    if fortran:
        return re.sub(r"!(?!\$omp)[^\n]*", blank, text, flags=re.I)
    return re.sub(r"//[^\n]*", blank, re.sub(r"/\*.*?\*/", blank, text, flags=re.S))


class Directive(object):
    def __init__(self, first, last, text, body, fortran):
        self.first, self.last, self.text, self.body, self.fortran = first, last, text, body, fortran
        self.masked = mask_comments(text, fortran)
        self.kind, self.names_end = self.classify()

    def words(self, limit=None, allowed=None):
        pat = WORD_F if self.fortran else WORD_C
        pos, found, ends = self.body, [], []
        while limit is None or len(found) < limit:
            m = pat.match(self.masked, pos)
            if not m:
                break
            word = m.group(1).lower()
            if allowed is not None and word not in allowed:
                break
            found.append(word)
            pos = m.end()
            ends.append(pos)
        return found, ends

    def classify(self):
        allowed = F_LOOP_WORDS if self.fortran else C_LOOP_WORDS
        loops = F_LOOPS if self.fortran else C_LOOPS
        names, ends = self.words(allowed=allowed)
        self.combined = bool(names) and names[0] == "parallel"
        if tuple(names) in loops:
            return "loop", ends[-1]
        first, _ = self.words(limit=2)
        w0 = first[0] if first else ""
        if w0 in ("parallel", "barrier"):
            return "reset", 0
        if self.fortran and w0 == "end" and len(first) > 1:
            if first[1] == "do":
                return "enddo", 0
            if first[1] == "parallel":
                return "reset", 0
        return "other", 0


def update_block_comment(line, in_block):
    i = 0
    while i < len(line):
        if in_block:
            j = line.find("*/", i)
            if j < 0:
                return True
            in_block, i = False, j + 2
        else:
            j, k = line.find("/*", i), line.find("//", i)
            if k >= 0 and (j < 0 or k < j):
                return False
            if j < 0:
                return False
            in_block, i = True, j + 2
    return in_block


def collect_c(lines):
    found, i, in_block = [], 0, False
    while i < len(lines):
        m = C_START.match(lines[i])
        if m and not in_block:
            j = i
            while j + 1 < len(lines) and lines[j].rstrip().endswith("\\"):
                j += 1
            found.append(Directive(i, j, "\n".join(lines[i:j + 1]), m.end(), False))
            i = j + 1
        else:
            in_block = update_block_comment(lines[i], in_block)
            i += 1
    return found


def collect_fortran(lines):
    found, i = [], 0
    while i < len(lines):
        m = F_START.match(lines[i])
        if not m:
            i += 1
            continue
        j = i
        while (j + 1 < len(lines) and F_START.match(lines[j + 1])
               and mask_comments(lines[j], True).rstrip().endswith("&")):
            j += 1
        found.append(Directive(i, j, "\n".join(lines[i:j + 1]), m.end(), True))
        i = j + 1
    return found


def find_schedule(d):
    """None, "irregular" or (start, end, kind) of the schedule clause."""
    m = SCHEDULE.search(d.masked)
    if not m:
        return None
    depth, i = 1, m.end()
    while i < len(d.masked) and depth:
        depth += (d.masked[i] == "(") - (d.masked[i] == ")")
        i += 1
    if depth or (d.fortran and "\n" in d.masked[m.start():i]):
        return "irregular"
    inner = SCHEDULE_MODIFIERS.sub("", d.masked[m.end():i - 1])
    kind = re.match(r"\s*(\w+)", inner)
    return (m.start(), i, kind.group(1).lower() if kind else "")


def has_nowait(dirs, idx):
    d = dirs[idx]
    if not d.fortran:
        return bool(NOWAIT.search(d.masked))
    if d.combined:
        return False
    for nxt in dirs[idx + 1:]:  # Fortran: nowait sits on the matching "end do"
        if nxt.kind == "enddo":
            return bool(NOWAIT.search(nxt.masked))
        if nxt.kind in ("loop", "reset"):
            break
    return False


def plan_edits(lines, fortran, allow_nowait):
    """Returns (edits, records). edits are (first, last, new_lines): physical lines first..last
    (0-based, inclusive) are replaced; records are (line, action, directive_first_line)."""
    dirs = collect_fortran(lines) if fortran else collect_c(lines)
    records, edits, chain = [], [], False
    for idx, d in enumerate(dirs):
        if d.kind == "reset":
            chain = False
            continue
        if d.kind != "loop":
            continue
        if d.combined:
            chain = False
        shown = lines[d.first].strip()[:110]
        sched = find_schedule(d)
        if sched == "irregular":
            records.append((d.first + 1, "skipped: unusual schedule clause", shown))
            continue
        nowait = has_nowait(dirs, idx)
        adaptive = sched is not None and sched[2] in ADAPTIVE
        unsafe = not adaptive and not allow_nowait
        skip = unsafe and (chain or nowait)
        if nowait:
            chain = chain or unsafe
        else:
            chain = False
        if skip:
            records.append((d.first + 1, "skipped: " + ("nowait loop" if nowait else "follows a nowait loop"), shown))
            continue
        if sched is not None and sched[2] == "runtime":
            records.append((d.first + 1, "already runtime", shown))
            continue

        upper = d.text[d.body:d.names_end].strip().isupper() if fortran else False
        clause = "schedule(runtime)"
        if upper:
            clause = clause.upper()
        if sched is None:
            new = d.text[:d.names_end] + " " + clause + d.text[d.names_end:]
            action = "added"
        else:
            new = d.text[:sched[0]] + clause + d.text[sched[1]:]
            action = "replaced schedule(%s)" % sched[2]
        new_lines = new.split("\n")
        if fortran and sched is None and any(len(x) > MAX_FORTRAN_LINE for x in new_lines):
            rest = d.text[d.names_end:].split("\n", 1)
            if rest[0].strip():  # put the clause on its own line and continue the directive
                new = (d.text[:d.names_end] + " " + clause + " &\n" + d.text[:d.body] + "& "
                       + rest[0].lstrip() + ("\n" + rest[1] if len(rest) > 1 else ""))
                new_lines = new.split("\n")
        if fortran and any(len(x) > MAX_FORTRAN_LINE for x in new_lines):
            records.append((d.first + 1, "skipped: line would exceed %d columns" % MAX_FORTRAN_LINE, shown))
            continue
        edits.append((d.first, d.last, new_lines))
        records.append((d.first + 1, action, shown))
    return edits, sorted(records)


def patch_text(text, fortran, allow_nowait):
    """Returns (new_text, records) for a text with "\\n" line endings."""
    lines = text.split("\n")
    edits, records = plan_edits(lines, fortran, allow_nowait)
    for first, last, new_lines in reversed(edits):
        lines[first:last + 1] = new_lines
    return "\n".join(lines), records


def split_lines(data):
    """Lines without terminator, plus each line's own terminator ("\\r\\n", "\\n" or "")."""
    lines, ends = [], []
    for raw in re.findall(r"[^\n]*\n|[^\n]+", data):
        end = "\r\n" if raw.endswith("\r\n") else "\n" if raw.endswith("\n") else ""
        lines.append(raw[:len(raw) - len(end)])
        ends.append(end)
    return lines, ends


def patch_file(path, fortran, allow_nowait, dry_run, backup):
    with open(path, "r", encoding="utf-8", errors="surrogateescape", newline="") as f:
        data = f.read()
    lines, ends = split_lines(data)
    edits, records = plan_edits(lines, fortran, allow_nowait)
    if edits and not dry_run:
        for first, last, new_lines in reversed(edits):  # untouched lines keep their own terminators
            ends[first:last + 1] = [ends[first]] * (len(new_lines) - 1) + [ends[last]]
            lines[first:last + 1] = new_lines
        if backup and not os.path.exists(path + ".orig"):
            shutil.copy2(path, path + ".orig")
        with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
            f.write("".join(line + end for line, end in zip(lines, ends)))
    return records


def source_files(root):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            yield os.path.join(dirpath, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="source directories (or files) to patch")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument("--backup", action="store_true", help="keep <file>.orig for files that are modified")
    ap.add_argument("--allow-nowait", action="append", default=[], metavar="PATH_SUBSTRING",
                    help="patch nowait loops in files whose path contains this text (reviewed loops only)")
    ap.add_argument("--exclude", action="append", default=[], metavar="PATH_SUBSTRING",
                    help="skip files whose path contains this text")
    ap.add_argument("--report", metavar="FILE", help="write every decision to this file")
    args = ap.parse_args()

    report, skipped_all = [], []
    print("%-34s %7s %9s %8s %8s" % ("root", "added", "replaced", "already", "skipped"))
    for root in args.roots:
        count = {"added": 0, "replaced": 0, "already": 0, "skipped": 0}
        for path in source_files(root):
            if any(x in path for x in args.exclude):
                continue
            suffix = os.path.splitext(path)[1].lower()
            if suffix in F_SUFFIXES:
                fortran = True
            elif suffix in C_SUFFIXES:
                fortran = False
            else:
                continue
            allow = any(a in path for a in args.allow_nowait)
            for line, action, shown in patch_file(path, fortran, allow, args.dry_run, args.backup):
                report.append("%s:%d: %s | %s" % (path, line, action, shown))
                key = ("added" if action == "added" else "replaced" if action.startswith("replaced")
                       else "already" if action == "already runtime" else "skipped")
                count[key] += 1
                if key == "skipped":
                    skipped_all.append("%s:%d: %s" % (path, line, action[len("skipped: "):]))
        print("%-34s %7d %9d %8d %8d" % (root, count["added"], count["replaced"], count["already"], count["skipped"]))

    if skipped_all:
        print("\nLeft unchanged (%d):" % len(skipped_all))
        for s in skipped_all:
            print("  " + s)
    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join(report) + "\n")
        print("\nFull report: " + args.report)
    if args.dry_run:
        print("(dry run: no file was modified)")


if __name__ == "__main__":
    main()
