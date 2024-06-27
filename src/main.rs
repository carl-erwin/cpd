// Copyright (c) Carl-Erwin Griffith
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// This is a simple text "parser" that looks for "almost" common ranges of lines.
// It only support ascii/utf8 files.

/*

  get all file_index, line_index for a given hash

  Key=<line_hash>  -> Set<(file_index, line_index)>

  for each elm in S



  File_Info = &file_info[file_index]

  File_Info.lines[line_index] -> (line_num, line_hash)


    elm.file_index, elm.line_index+1 (?)





*/

extern crate clap;
extern crate crc;
extern crate num_cpus;
extern crate walkdir;

use walkdir::WalkDir;

use clap::{App, Arg};

use crc::{crc64, Hasher64};

use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;

const DEFAULT_MIN_WINDOW_SIZE: usize = 3;

const DEFAULT_STACK_SIZE: usize = 64 * 1024 * 1024;
const MIN_STACK_SIZE: usize = 2 * 4096;

const VERSION: &str = env!("CARGO_PKG_VERSION");

struct Config {
    nr_threads: usize,
    stack_size: usize,
    min_window_size: usize,
    files: Vec<String>,
    print_results: bool,
    parse_only: bool,
}

type FileIndex = u32;
type LineIndex = u32;
type LineNumber = u32;
type CpdHash = u64;

type CpdMap = HashMap<CpdHash, HashSet<(FileIndex, LineIndex)>>;

#[derive(Debug, Copy, Clone)]
struct WindowSize(usize);

#[derive(Debug, Copy, Clone)]
struct CallDepth(u32);

#[derive(Debug, Copy, Clone)]
struct LineInfo {
    line_number: usize,
    hash: CpdHash,
}

impl LineInfo {
    fn new(line_number: usize, hash: CpdHash) -> LineInfo {
        LineInfo { line_number, hash }
    }
}

#[derive(Debug)]
struct FileInfo {
    filename: String,
    lines: Vec<LineInfo>,
}

struct CpdResults {
    window_to_files_range: CpdMap,
    window_size_map: HashMap<u32, HashSet<CpdHash>>,
}

impl CpdResults {
    fn new() -> CpdResults {
        CpdResults {
            window_to_files_range: HashMap::new(),
            window_size_map: HashMap::new(),
        }
    }
}

/// This function computes a 64 bits crc for each file's line.
/// It will ignore empty lines and some blank characters (' ', '\t', '\n').
/// Each crc serves as a key in a shared map to store a (file_index, line_index) tuple.
/// The (file_index, line_index) tuple is stored in a HashSet and then in the hash set is stored in the global map.
///
/// Summary: hashmap[crc] -> HashSet<(file_index, line_index)>
fn parse_file(filename: &str, file_index: usize, map: &RwLock<CpdMap>) -> Option<Vec<LineInfo>> {
    let file = File::open(filename);
    let file = match file {
        Ok(file) => file,
        Err(e) => {
            eprintln!("cannot parse {}, {:?}", filename, e);
            return None;
        }
    };

    let file_index = file_index as FileIndex;
    let mut line_info = Vec::<LineInfo>::new();

    // 1 - read line
    let buf_reader = BufReader::new(file);
    let lines = buf_reader.lines();

    let mut map = map.write().unwrap();

    for (line_number, l) in lines.enumerate() {
        let l = match l {
            Ok(l) => l,
            Err(_) => break,
        };

        let mut digest = crc64::Digest::new(crc64::ECMA);

        // NB: String internal representation is utf8
        // we can filter instead
        // ignore  spaces|tabs|newline
        let mut slen = 0;
        for c in l.into_bytes() {
            let sl: [u8; 1] = [c];
            match sl[0] {
                b'\0'..=b' ' | b'{' | b'}' | b'(' | b')' => {}
                _ => {
                    slen += 1;
                    digest.write(&sl);
                }
            }
        }

        // ignore empty lines
        if slen <= 0 {
            continue;
        }

        // compute line checksum
        let hash = digest.sum64();

        // build line info
        let line_index = line_info.len() as LineIndex;
        line_info.push(LineInfo::new(line_number + 1, hash));

        // map[hash] += HashSetEntry<(file_index, line_index)>
        map.entry(hash)
            .or_insert_with(HashSet::new)
            .insert((file_index, line_index));
    }

    Some(line_info)
}

/// This functions runs multiple threads to call parse_file(...) .
/// It returns all hash and the shared map that holds all crc -> (file_index, file_line) mapping.
fn parse_files(
    n_jobs: usize,
    file_max: usize,
    files_inf: &Arc<RwLock<Vec<FileInfo>>>,
) -> (Vec<CpdHash>, Vec<Arc<RwLock<CpdMap>>>) {
    let mut thread_map = Vec::with_capacity(n_jobs);

    for _ in 0..n_jobs {
        thread_map.push(Arc::new(RwLock::new(CpdMap::new())));
    }

    let mut handles = vec![];
    let idx = Arc::new(AtomicUsize::new(0));

    for job_idx in 0..n_jobs {
        let files_inf = Arc::clone(&files_inf);
        let idx = Arc::clone(&idx);

        let sub_graph = Arc::clone(&thread_map[job_idx]);

        let handle = thread::spawn(move || loop {
            let i = idx.fetch_add(1, Ordering::SeqCst);
            if i >= file_max {
                return;
            }

            let filename = {
                let vec = files_inf.read().unwrap();
                vec[i].filename.clone()
            };

            eprint!("\rparse files : {}/{}               ", i + 1, file_max);

            if let Some(lines) = parse_file(&filename, i, &sub_graph) {
                let mut vec = files_inf.write().unwrap();
                vec[i].lines = lines;
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let mut hash_vec = Vec::<CpdHash>::new();

    // keys to vec
    for map in thread_map.iter() {
        let hash_graph = map.read().unwrap();
        hash_vec.extend((*hash_graph).keys());
    }

    eprintln!("");

    (hash_vec, thread_map)
}

/// This function prints a specific range of line, for a given file
fn print_file_lines(filename: &str, start_line: usize, end_line: usize) {
    let file = File::open(filename);
    let file = match file {
        Ok(file) => file,
        _ => {
            eprintln!("cannot print {}", filename);
            return;
        }
    };

    let reader = BufReader::new(file);
    let mut writer = BufWriter::new(io::stdout());

    for line in reader
        .lines()
        .skip(start_line - 1)
        .take(end_line - start_line + 1)
    {
        let l = line.unwrap();
        writeln!(writer, "{}", l).unwrap();
    }
}


// TODO:(ceg) find a way to merge hash maps
// to avoid memory explosion

/// This (recursive) function compares the consecutive crcs to detect the cut/paste code.
/// for a given starting crc there is a list/array of (file_index,line_index) tuples.
/// We check all (file_index, line_index + 1).
/// If they all point to the same crc we loop  (cut/paste detected).
/// If any next line diverges, we store the current results (if the minimum number of line reached).
/// We recurse with all "new" crcs until there is no next line.
fn walk_graph(
    results: &Arc<RwLock<CpdResults>>,
    min_window_size: usize,
    window_size: WindowSize,
    depth: CallDepth,
    files_inf: &[FileInfo],
    hash_graph: &Arc<Vec<Arc<RwLock<CpdMap>>>>,
    current_lines: &mut Vec<(CpdHash, FileIndex, LineIndex)>,
) {
    let mut window_size = window_size;

    loop {
        window_size.0 += 1;

        let mut to_flush = false;
        let mut next_lines: Vec<(CpdHash, FileIndex, LineIndex)> =
            Vec::with_capacity(current_lines.len());

        let mut sub_graph: HashMap<CpdHash, HashSet<(u32, u32)>> = HashMap::new();

        // TODO: fn compute_next_lines(&hash_graph, &mut to_flush, &mut next_lines)
        // build vec from current_lines<(fi,li)> ->  next_lines<(fi,li+1)>
        for e in current_lines.iter() {
            let finfo = &files_inf[e.1 as usize];

            let next_li = (e.2 + 1) as usize;

            // reached end-of-file ?
            if next_li >= finfo.lines.len() {
                if window_size.0 >= min_window_size && current_lines.len() > 1 {
                    // flush only if there are at least two files
                    to_flush = true;
                }

                // eof reached process next file
                continue;
            }

            // get next line hash
            let next_hash = finfo.lines[next_li].hash;

            // TODO(ceg): wallk-through all sub graph
            for hash_graph in hash_graph.iter() {
                let hash_graph = hash_graph.read().unwrap();
                if let Some(lines) = hash_graph.get(&next_hash) {
                    if lines.len() > 1 {
                        // at least 2 lines
                        next_lines.push((next_hash, e.1, next_li as u32));

                        // we must keep track of the next lines and build a new graph
                        // if there are more than two hashes we will recurse
                        sub_graph
                            .entry(next_hash)
                            .or_insert_with(HashSet::new)
                            .insert((e.1, next_li as u32));
                    } else if window_size.0 >= min_window_size && current_lines.len() > 1 {
                        // unique line detected: must flush , restart window
                        // restart window for this index
                        // when new file(s) are detected must insert new window
                        to_flush = true;
                    }
                } else if window_size.0 >= min_window_size && current_lines.len() > 1 {
                    // unique line detected: must flush , restart window
                    // restart window for this index
                    // when new file(s) are detected must insert new window
                    to_flush = true;
                }
            }
        }

        // will recurse
        if sub_graph.len() > 1 && window_size.0 >= min_window_size {
            to_flush = true;
        }

        // compute result before recursion
        if to_flush {
            // build hash of all hashes
            // insert in 2 maps
            // window_size to hset(window_hash) to hset(fi, li) : u32 -> CpdHash, hmap(CpdHash) -> hset((FileIndex, LineNumber))

            let hash_concat = {
                let window_size = window_size.0 as u32;
                let mut window_digest = crc64::Digest::new(crc64::ECMA);

                let e = &current_lines[0];
                let (fi, li) = (e.1 as usize, e.2 as usize);
                let finfo = &files_inf[fi];

                for i in 0..(window_size as usize) {
                    let h = finfo.lines[i + (li + 1) - (window_size as usize)].hash;
                    let bytes = h.to_be_bytes();
                    window_digest.write(&bytes);
                }

                window_digest.sum64()
            };

            // insert window hash
            {
                let window_size = window_size.0 as u32;

                results
                    .write()
                    .unwrap()
                    .window_size_map
                    .entry(window_size)
                    .or_insert_with(HashSet::new)
                    .insert(hash_concat);
            }

            for e in current_lines.iter() {
                let (fi, li) = (e.1 as usize, e.2 as usize);

                // NB: "(li + 1) - window_size.0"
                // and not "li - window_size.0 + 1" to avoid underflow
                let l_start_index = (li + 1) - window_size.0;

                results
                    .write()
                    .unwrap()
                    .window_to_files_range
                    .entry(hash_concat)
                    .or_insert_with(HashSet::new)
                    .insert((fi as FileIndex, l_start_index as LineNumber));
            }
        }

        // unique hash ? loop
        if sub_graph.len() == 1 {
            // prepare next cursors
            *current_lines = next_lines.clone();
            continue;
        }

        // for each "new" crc
        for (hash, hash_set) in &sub_graph {
            let mut current_lines: Vec<(CpdHash, FileIndex, LineIndex)> = Vec::new();
            for set in hash_set.iter() {
                current_lines.push((*hash, set.0, set.1));
            }

            walk_graph(
                results,
                min_window_size,
                WindowSize(window_size.0),
                CallDepth(depth.0 + 1),
                &files_inf,
                &hash_graph,
                &mut current_lines,
            );
        }

        break;
    }
}

/// This function detects the starting point of a cut/paste region.
/// It simply compares the crcs of each previous lines.
/// If it diverges (or some previous lines cannot be computed): you have your starting point.
fn get_first_common_line(
    files_inf: &[FileInfo],
    current_lines: &mut Vec<(CpdHash, FileIndex, LineIndex)>,
) {
    let mut previous_lines: Vec<(CpdHash, FileIndex, LineIndex)> = Vec::new();

    if current_lines.is_empty() {
        return;
    }

    loop {
        let mut prev_hash: CpdHash = 0;
        for (index, l) in current_lines.iter().enumerate() {
            let &(_, fi, li) = l;

            if li == 0 {
                // cannot compute previous line
                return;
            }

            let finfo = &files_inf[fi as usize];
            let prev_li = li - 1;
            let hash = finfo.lines[prev_li as usize].hash;

            if index > 0 && hash != prev_hash {
                // found an other hash
                return;
            }

            prev_hash = hash;
            previous_lines.push((prev_hash, fi, prev_li));
        }

        current_lines.clear();
        current_lines.append(&mut previous_lines);
    }
}

/// walkthrough all hashes and rewind to find the first common hash
fn filter_common_starting_point(
    n_jobs: usize,
    hash_vec: Vec<CpdHash>,
    files_inf: &Arc<RwLock<Vec<FileInfo>>>,
    hash_graph: &Arc<Vec<Arc<RwLock<CpdMap>>>>,
) -> Vec<CpdHash> {
    let nr_hash = hash_vec.len();

    let hash_vec = Arc::new(hash_vec);
    let idx = Arc::new(AtomicUsize::new(0));
    let idx = Arc::clone(&idx);

    let filtered_hash = Arc::new(RwLock::new(HashSet::<CpdHash>::new()));

    let mut handles = vec![];

    eprintln!("filtering starting points...");

    for th_idx in 0..n_jobs {
        let idx = Arc::clone(&idx);
        let files_inf = Arc::clone(&files_inf);
        let hash_vec = Arc::clone(&hash_vec);
        let hash_graph = Arc::clone(&hash_graph);
        let filtered_hash = Arc::clone(&filtered_hash);

        let th_name = format!("hash_filter_{}", th_idx);
        let builder = thread::Builder::new().name(th_name);

        let handle = builder
            .spawn(move || loop {
                let i = idx.fetch_add(1, Ordering::SeqCst);
                if i >= nr_hash {
                    return;
                }

                let files_inf = files_inf.read().unwrap();
                let hash = hash_vec[i];

                let mut current_lines: Vec<(CpdHash, FileIndex, LineIndex)> = Vec::new();

                // iter over all sub graph
                for sub_graph in hash_graph.as_ref().iter() {
                    let sub_graph = sub_graph.read().unwrap();
                    if let Some(hash_set) = sub_graph.get(&hash) {
                        for e in hash_set.iter() {
                            current_lines.push((hash, e.0, e.1));
                        }
                    }
                }

                get_first_common_line(&files_inf, &mut current_lines);

                let hash = current_lines[0].0;
                filtered_hash.write().unwrap().insert(hash);
            })
            .unwrap();
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    eprintln!("done");

    let filtered_hash = filtered_hash.read().unwrap();
    let mut hash_vec = Vec::with_capacity(filtered_hash.len());

    eprintln!(
        "found {} common starting points...",
        nr_hash - filtered_hash.len()
    );

    // HashSet -> Vec
    hash_vec.extend(filtered_hash.iter());
    hash_vec
}

fn parse_graph(
    n_jobs: usize,
    stack_size: usize,
    hash_graph: &Arc<Vec<Arc<RwLock<CpdMap>>>>,
    hash_vec: Vec<CpdHash>,
    files_inf: &Arc<RwLock<Vec<FileInfo>>>,
    results: &Arc<RwLock<CpdResults>>,
    min_window_size: usize,
) {
    eprintln!("parsing graph nodes...");
    let nr_hash = hash_vec.len();
    let hash_vec = Arc::new(RwLock::new(hash_vec));

    let idx = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];
    for th_idx in 0..n_jobs {
        let results = Arc::clone(&results);
        let idx = Arc::clone(&idx);
        let hash_graph = Arc::clone(&hash_graph);
        let hash_vec = Arc::clone(&hash_vec);
        let files_inf = Arc::clone(files_inf);

        let th_name = format!("worker_{}", th_idx);
        let builder = thread::Builder::new().name(th_name).stack_size(stack_size);

        let handle = builder
            .spawn(move || loop {
                let i = idx.fetch_add(1, Ordering::SeqCst);
                if i >= nr_hash {
                    return;
                }

                let mut current_lines_set: HashSet<(CpdHash, FileIndex, LineIndex)> =
                    HashSet::new();

                let files_inf = files_inf.read().unwrap();
                let hash_vec = hash_vec.read().unwrap();
                let hash = hash_vec[i];

                // get all (file,line_indexes) that match hash
                for hash_graph in hash_graph.iter() {
                    let hash_graph = hash_graph.read().unwrap();
                    if let Some(hash_set) = hash_graph.get(&hash) {
                        // eprintln!("hash 0x{:x} {{", hash);
                        for set in hash_set.iter() {
                            current_lines_set.insert((hash, set.0, set.1));
                        }
                    }
                }

                // HashSet -> Vec
                let mut current_lines: Vec<(CpdHash, FileIndex, LineIndex)> = Vec::new();
                current_lines.extend(current_lines_set.iter());
                current_lines_set.clear();

                println!("current_lines len = {}", current_lines.len());

                walk_graph(
                    &results,
                    min_window_size,
                    WindowSize(0),
                    CallDepth(1),
                    &files_inf,
                    &hash_graph,
                    &mut current_lines,
                );
            })
            .unwrap();
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    eprintln!("done");
}

fn print_results(files_inf: &RwLock<Vec<FileInfo>>, results: &RwLock<CpdResults>) {
    // add option : print results
    let files_inf = files_inf.read().unwrap();
    let map = &results.read().unwrap().window_size_map;
    let mut window_hash_vec = Vec::<(u32, &HashSet<CpdHash>)>::new();

    // map -> vec
    for (k, v) in map {
        window_hash_vec.push((*k, v));
    }
    // sort by window size greater first
    window_hash_vec.sort_unstable_by(|a, b| (b.0).cmp(&a.0));

    let file_map = &results.read().unwrap().window_to_files_range;

    // print all windows
    for w in &window_hash_vec {
        let (window_size, cpd_hashes) = (w.0 as usize, w.1);

        // sort hashes
        let mut h_vec = Vec::new();
        h_vec.extend(cpd_hashes.into_iter());
        h_vec.sort();

        for hash in h_vec.iter() {
            println!("\n\nfound window size {}", window_size);
            println!("checking  {:?}", hash);

            if let Some(hset) = file_map.get(hash) {
                // sort file by file index
                let mut v = Vec::new();
                for (fi, li) in hset.iter() {
                    v.push((*fi as usize, *li as usize));
                }

                v.sort_unstable_by(|a, b| {
                    let res = (a).0.cmp(&b.0);
                    if res == std::cmp::Ordering::Equal {
                        (a).1.cmp(&b.1)
                    } else {
                        res
                    }
                });

                println!("number of files : {}", v.len());
                for (index, (fi, li)) in v.iter().enumerate() {
                    let finfo = &files_inf[*fi];
                    let l_start = finfo.lines[*li].line_number;
                    let l_end = finfo.lines[*li + window_size - 1].line_number;

                    println!("{} : {}-{}", finfo.filename, l_start, l_end);

                    if index + 1 == v.len() {
                        println!("8<----------------");
                        print_file_lines(&finfo.filename, l_start, l_end);
                        println!("8<----------------");
                    }
                }
            }
        }
    }
}

fn read_file(filename: &str) -> Vec<String> {
    let mut v = Vec::new();
    let file = File::open(filename);
    let file = match file {
        Ok(file) => file,
        _ => {
            eprintln!("cannot parse {}", filename);
            return v;
        }
    };

    let buf_reader = BufReader::new(file);
    for line in buf_reader.lines() {
        v.push(line.unwrap());
    }

    v
}

fn parse_command_line() -> Config {
    let matches = App::new("cpd")
        .version(VERSION)
        .author("Carl-Erwin Griffith <carl.erwin@gmail.com>")
        .about("simple cut/paste detector")
        .arg(
            Arg::with_name("THREAD")
                .short("t")
                .long("thread")
                .value_name("NUMBER")
                .help("sets number of thread to scan the files")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("STACK_SIZE")
                .short("s")
                .long("stack_size")
                .value_name("STACK_SIZE")
                .help("sets worker thread stack size in bytes (default Mib)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("FL")
                .short("l")
                .long("list")
                .value_name("FL")
                .help("this file contains the list of files to scan")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("WINDOW")
                .short("w")
                .long("min_window")
                .value_name("WINDOW")
                .help("sets the minimun number of lines to consider in results")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("PS")
                .short("p")
                .long("--parse-only")
                .help("stop after indexing"),
        )
        .arg(
            Arg::with_name("PR")
                .short("r")
                .long("--skip-results")
                .help("do not print results"),
        )
        .arg(
            Arg::with_name("DIRS")
                .short("d")
                .long("--dir")
                .help("list of directories to scan")
                .takes_value(true)
                .required(false)
                .multiple(true),
        )
        .arg(
            Arg::with_name("FILES")
                .help("list of the files to scan")
                .required(false)
                .multiple(true),
        )
        .get_matches();

    let nr_threads = if matches.is_present("THREAD") {
        let nr_threads = matches.values_of("THREAD").unwrap().collect::<String>();
        nr_threads.trim_end().parse::<usize>().unwrap_or(1)
    } else {
        num_cpus::get()
    };

    let nr_threads = ::std::cmp::max(1, nr_threads);

    let stack_size = if matches.is_present("STACK_SIZE") {
        let stack_size = matches.values_of("STACK_SIZE").unwrap().collect::<String>();
        stack_size
            .trim_end()
            .parse::<usize>()
            .unwrap_or(DEFAULT_STACK_SIZE)
    } else {
        DEFAULT_STACK_SIZE
    };

    let stack_size = ::std::cmp::max(MIN_STACK_SIZE, stack_size);

    let min_window_size = if matches.is_present("WINDOW") {
        let min_window_size = matches.values_of("WINDOW").unwrap().collect::<String>();
        min_window_size
            .trim_end()
            .parse::<usize>()
            .unwrap_or(DEFAULT_MIN_WINDOW_SIZE)
    } else {
        DEFAULT_MIN_WINDOW_SIZE
    };

    let mut files = Vec::new();

    if matches.is_present("FL") {
        let filename = matches.values_of("FL").unwrap().collect::<String>();
        files.extend(read_file(&filename));
    }

    if matches.is_present("FILES") {
        let list: Vec<String> = matches
            .values_of("FILES")
            .unwrap()
            .map(|x| x.to_owned())
            .collect();

        files.extend(list);
    }

    let mut dir_list = Vec::new();
    if files.is_empty() {
        dir_list.push(".".to_owned());
    }

    if matches.is_present("DIRS") {
        let mut list: Vec<String> = matches
            .values_of("DIRS")
            .unwrap()
            .map(|x| x.to_owned())
            .collect();

        dir_list.append(&mut list);
    }

    for d in dir_list {
        println!("scanning {}", d);

        for entry in WalkDir::new(d) {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.is_dir() {
                // println!("ignore dir {}", path.display());
                continue;
            }

            if path.is_symlink() {
                //println!("ignore symlink {}", path.display());
                continue;
            }

            files.push(path.as_os_str().to_str().unwrap().to_owned());

            //                println!("{}", entry.path().display());
        }
    }

    let print_results = !matches.is_present("PR");
    let parse_only = matches.is_present("PS");

    Config {
        nr_threads,
        files,
        min_window_size,
        stack_size,
        print_results,
        parse_only,
    }
}

fn main() {
    let config = parse_command_line();

    // dedup files
    let set: HashSet<String> = config.files.into_iter().collect();
    let mut args: Vec<String> = set.into_iter().collect();

    // sort files list
    args.sort();

    let results = Arc::new(RwLock::new(CpdResults::new()));

    // prepare file slots
    let mut v = Vec::<FileInfo>::with_capacity(args.len());
    for (_i, item) in args.iter().enumerate() {
        v.push(FileInfo {
            //file_index: i,
            filename: item.clone(),
            lines: Vec::new(),
        });
    }
    let files_inf = Arc::new(RwLock::new(v));

    let (hash_vec, hash_graph) = parse_files(config.nr_threads, args.len(), &files_inf);

    if config.parse_only {
        return;
    }

    let hash_graph = Arc::new(hash_graph);
    let hash_vec =
        filter_common_starting_point(config.nr_threads, hash_vec, &files_inf, &hash_graph);

    parse_graph(
        config.nr_threads,
        config.stack_size,
        &hash_graph,
        hash_vec,
        &files_inf,
        &results,
        config.min_window_size,
    );

    if config.print_results {
        print_results(&files_inf, &results);
    }
}
