#pragma once

// Input checks shared by fastibs, fastibsmapper and KDBIntersect: the
// command-line paths and the window size are checked up front, so a typo
// ends in one "Error: ..." line and exit status 1 instead of an uncaught
// std::filesystem_error (abort, status 134), a division by zero (W = 0) or
// an endless window loop (W <= k). Nothing here touches a valid input:
// the same folders, files and window sizes are accepted and resolved as
// before. Free of KMC and CUDA so the host tests can include it.

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace cli
{
namespace fs = std::filesystem;

// -h / --help anywhere on the command line: the caller prints its usage
// on stdout and exits 0.
inline bool helpRequested(int argc, char *argv[])
{
    for (int a = 1; a < argc; a++)
    {
        const std::string arg = argv[a];
        if (arg == "-h" || arg == "--help")
            return true;
    }
    return false;
}

// --version anywhere on the command line: the caller prints versionLine()
// on stdout and exits 0 (checked before --help and the positional
// arguments, like GNU tools).
inline bool versionRequested(int argc, char *argv[])
{
    for (int a = 1; a < argc; a++)
        if (std::string(argv[a]) == "--version")
            return true;
    return false;
}

#ifndef GPUIBS_VERSION
#define GPUIBS_VERSION "unknown"
#endif
#ifndef GPUIBS_REVISION
#define GPUIBS_REVISION ""
#endif

// "<tool> (GPU-IBS) <version> [git <revision>], GPU and CPU build" -- what a
// pipeline records to say which build produced its results. CMake defines
// GPUIBS_VERSION (the project version) and GPUIBS_REVISION (git describe of
// the checkout, empty outside a clone).
inline std::string versionLine(const char *tool)
{
    std::string s = std::string(tool) + " (GPU-IBS) " + GPUIBS_VERSION;
    if (GPUIBS_REVISION[0] != '\0')
        s += std::string(" (git ") + GPUIBS_REVISION + ")";
#ifdef FASTIBS_CUDA
    s += ", GPU and CPU build";
#else
    s += ", CPU-only build";
#endif
    return s;
}

// A whole decimal integer: what std::stoi accepted (leading blanks, a
// sign, leading zeros) but nothing after the digits ("50000abc", "5e4" and
// "" are rejected) and nothing outside [lo, INT_MAX].
inline bool parseInt(const std::string &text, int lo, int &out)
{
    const char *s = text.c_str();
    char *end = nullptr;
    errno = 0;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || v < lo || v > INT_MAX)
        return false;
    out = (int)v;
    return true;
}

// <windowSize> as typed: a positive integer. Whether it is also larger
// than k is checked once the database header is read (windowFitsK).
inline bool parseWindowSize(const std::string &text, int &out)
{
    return parseInt(text, 1, out);
}

// Consecutive windows overlap by k bases (chunkSequences steps by W - k),
// so a window must be longer than k: W == k never advances (the original
// tool loops until memory runs out), W < k wraps the unsigned step.
inline bool windowFitsK(long long windowSize, unsigned k)
{
    return windowSize > (long long)k;
}

inline void reportWindowTooSmall(int windowSize, unsigned k)
{
    std::cerr << "Error: <windowSize> " << windowSize << " must be larger than the database's k ("
              << k << "): consecutive windows overlap by k bases, so W > k is required" << std::endl;
}

// The KMC prefix in a dataset folder, as the original findPrefix found it:
// the name before ".kmc_" of the first such file. "" when there is none
// or the folder cannot be listed.
inline std::string findPrefix(const std::string &dir)
{
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
    {
        const std::string filename = it->path().filename().string();
        const size_t pos = filename.find(".kmc_");
        if (pos != std::string::npos)
            return filename.substr(0, pos);
    }
    return "";
}

// A KMC dataset folder -> "<dir>/<prefix>" (what the tools open). False,
// after one error line naming the path, when the folder is missing, is
// not a folder, or holds no <prefix>.kmc_pre + <prefix>.kmc_suf pair. A
// KMC prefix given in place of its folder gets a hint.
inline bool resolveDatabase(const std::string &dir, std::string &prefixPath)
{
    std::error_code ec;
    if (!fs::is_directory(dir, ec))
    {
        std::cerr << "Error: KMC database folder not found: " << dir;
        if (fs::exists(dir + ".kmc_pre", ec))
            std::cerr << " (this is a KMC prefix; pass the folder that holds "
                      << fs::path(dir).filename().string() << ".kmc_pre / .kmc_suf)";
        std::cerr << std::endl;
        return false;
    }
    const std::string prefix = findPrefix(dir);
    const std::string path = dir + "/" + prefix;
    if (prefix.empty() || !fs::exists(path + ".kmc_pre", ec) || !fs::exists(path + ".kmc_suf", ec))
    {
        std::cerr << "Error: no KMC database (<prefix>.kmc_pre and <prefix>.kmc_suf) in " << dir
                  << std::endl;
        return false;
    }
    prefixPath = path;
    return true;
}

// Reference files: *.fasta (the original tool's only extension), *.fa and
// *.fna, each optionally gzip-compressed (*.fasta.gz, ...; readSequenceFile
// decompresses them with zlib).
inline std::string stripGz(const std::string &filename)
{
    const size_t n = filename.size();
    if (n > 3 && filename.compare(n - 3, 3, ".gz") == 0)
        return filename.substr(0, n - 3);
    return filename;
}

inline bool isReferenceFile(const std::string &filename)
{
    const std::string ext = fs::path(stripGz(filename)).extension().string();
    return ext == ".fasta" || ext == ".fa" || ext == ".fna";
}

inline const char *referenceKinds()
{
    return ".fasta, .fa or .fna, optionally .gz";
}

// The reference files of a folder, in directory order (as the original
// iterated it); every other entry is named on stdout as skipped. False,
// after one error line, when the folder cannot be listed or holds no
// reference file at all (the original then exited 0 having written
// nothing).
inline bool listReferences(const std::string &dir, std::vector<fs::path> &out)
{
    out.clear();
    std::error_code ec;
    if (!fs::is_directory(dir, ec))
    {
        std::cerr << "Error: reference folder not found: " << dir << std::endl;
        return false;
    }
    fs::directory_iterator it(dir, ec), end;
    for (; !ec && it != end; it.increment(ec))
    {
        const std::string name = it->path().filename().string();
        if (isReferenceFile(name))
            out.push_back(it->path());
        else
            std::cout << "Skipping " << name << " (not a reference file: " << referenceKinds()
                      << ")" << std::endl;
    }
    if (ec)
    {
        std::cerr << "Error: cannot list the reference folder " << dir << ": " << ec.message()
                  << std::endl;
        return false;
    }
    if (out.empty())
    {
        std::cerr << "Error: no reference files (" << referenceKinds() << ") in " << dir
                  << std::endl;
        return false;
    }
    return true;
}

// The results folder, created (with its parents) when missing. False,
// after one error line, when it cannot be created or is not a folder.
inline bool prepareResultsFolder(const std::string &dir)
{
    std::error_code ec;
    if (fs::is_directory(dir, ec))
        return true;
    if (fs::exists(dir, ec))
    {
        std::cerr << "Error: results path exists and is not a folder: " << dir << std::endl;
        return false;
    }
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir))
    {
        std::cerr << "Error: cannot create the results folder " << dir << ": "
                  << (ec ? ec.message() : "not a folder") << std::endl;
        return false;
    }
    return true;
}

} // namespace cli
