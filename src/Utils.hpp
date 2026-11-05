#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <chrono>
#include <tuple>
#include <zlib.h>
#include <cstdlib>
#include <fstream>

#include "KmerCodec.hpp"

#define ZIP_BUFFER_SIZE (1 << 20)

using namespace std;

enum class FileType
{
    FASTA,
    FASTQ
};

class ParseResult
{
public:
    vector<string> ids;
    vector<string> sequences;
};


string reverseComplement(const string &kmer)
{
    string revComplement;
    revComplement.reserve(kmer.size());
    for (auto it = kmer.rbegin(); it != kmer.rend(); ++it)
    {
        switch (*it)
        {
        case 'A':
            revComplement += 'T';
            break;
        case 'C':
            revComplement += 'G';
            break;
        case 'G':
            revComplement += 'C';
            break;
        case 'T':
            revComplement += 'A';
            break;           
        default:
            throw invalid_argument("Invalid base in kmer");
        }
    }
    return revComplement;
}

string convertIntToKmer(uint64_t kmerInt, int kmerLength)
{
    // Convert an integer to a kmer
    string kmer(kmerLength, 'A'); // Initialize a kmer string with 'A's
    for (int i = kmerLength - 1; i >= 0; --i)
    {
        switch (kmerInt & 3) // Get the last two bits
        {
        case 0:
            kmer[i] = 'A';
            break;
        case 1:
            kmer[i] = 'C';
            break;
        case 2:
            kmer[i] = 'G';
            break;
        case 3:
            kmer[i] = 'T';
            break;
        }
        kmerInt >>= 2; // Shift the kmerInt right by 2 bits
    }
    return kmer;
}

// Canonical k-mers as 2-bit packed 64-bit codes via the rolling codec;
// invalid positions (windows covering non-ACGT bytes) are skipped, matching
// the compaction of the legacy string version below.
vector<uint64_t> getCanonicalKmers(const string &sequence, int kmerSize)
{
    vector<uint64_t> codes;
    KmerCodec(kmerSize).getCanonicalCodes(sequence, codes);
    return codes;
}

// Legacy string-based extraction, kept as the correctness oracle.
vector<string> getCanonicalKmersLegacy(string &sequence, int kmerSize)
{
    vector<string> kmers;
    if (sequence.size() < kmerSize)
        return kmers;
    transform(sequence.begin(), sequence.end(), sequence.begin(), ::toupper);
    for (int i = 0; i <= sequence.size() - kmerSize; i++)
    {
        try
        {
            string kmer = sequence.substr(i, kmerSize);
            if (kmer.find_first_not_of("ACGT") != string::npos)
                continue;
            string revComplement = reverseComplement(kmer);
            string canonicalKmer = kmer > revComplement ? revComplement : kmer;
            kmers.push_back(canonicalKmer);
        }
        catch (invalid_argument &e)
        {
            continue;
        }
    }
    return kmers;
}

string decompressGzip(const string &filename)
{
    cout << "Decompressing " << filename << endl;
    gzFile file = gzopen(filename.c_str(), "rb");
    if (!file)
    {
        throw runtime_error("Failed to open gzip file " + filename);
    }
    gzbuffer(file, ZIP_BUFFER_SIZE);
    vector<char> buffer(ZIP_BUFFER_SIZE);
    string decompressedData;

    while (true)
    {
        int bytesRead = gzread(file, buffer.data(), buffer.size());
        // A corrupt or truncated stream is an error, not a shorter
        // reference. The error state must be read after EVERY gzread: a
        // stream that ends inside a gzip member makes the read that hits
        // the end return the bytes it still decoded and set Z_BUF_ERROR
        // ("unexpected end of file"), and zlib 1.3.2 clears that error on
        // the next gzread (1.2.x and 1.3.1 keep it), so a check at EOF
        // alone accepts a truncated file with the newer zlib.
        int errnum = Z_OK;
        const char *why = gzerror(file, &errnum);
        if (bytesRead < 0 || (errnum != Z_OK && errnum != Z_STREAM_END))
        {
            const string reason = errnum == Z_BUF_ERROR ? "truncated gzip stream"
                                                        : string(why ? why : "read error");
            gzclose(file);
            throw runtime_error("Failed to decompress " + filename + ": " + reason);
        }
        if (bytesRead == 0)
            break;

        decompressedData.append(buffer.data(), bytesRead);
    }

    gzclose(file);
    return decompressedData;
}



ParseResult readSequenceFile(const string &filename, bool compressed = false)
{
    string data;
    if (compressed)
        data = decompressGzip(filename);
    else
    {
        ifstream file(filename, ios::binary | ios::ate);
        if (!file)
            throw runtime_error("Failed to open file " + filename);
        data.resize(static_cast<size_t>(file.tellg()));
        file.seekg(0);
        file.read(data.data(), data.size());
    }
    FileType fileType;

    if (!data.empty())
    {
        if (data[0] == '>')
            fileType = FileType::FASTA;
        else if (data[0] == '@')
            fileType = FileType::FASTQ;
        else
            throw runtime_error("Unknown file type");
    }
    else
        throw runtime_error("Empty file");

    cout << "Parsing " << (fileType == FileType::FASTA ? "fasta" : "fastq") << " data\n";
    ParseResult result;

    // Scan lines in place (same semantics as getline: split on '\n' only,
    // final partial line included) — avoids re-buffering the whole file
    // through a stringstream.
    string currentID;
    string currentSeq;
    bool isQualityLine = false;
    const char idChar = (fileType == FileType::FASTA ? '>' : '@');
    size_t pos = 0;
    while (pos < data.size())
    {
        size_t eol = data.find('\n', pos);
        if (eol == string::npos)
            eol = data.size();
        const char *line = data.data() + pos;
        const size_t lineLen = eol - pos;
        pos = eol + 1;

        if (lineLen == 0)
            continue;

        if (fileType == FileType::FASTQ && line[0] == '+')
        {
            isQualityLine = true;
            continue;
        }

        if (isQualityLine)
        {
            isQualityLine = false;
            continue;
        }

        if (line[0] == idChar)
        {
            if (!currentSeq.empty())
            {
                result.ids.push_back(std::move(currentID));
                result.sequences.push_back(std::move(currentSeq));
                currentID.clear();
                currentSeq.clear();
            }
            currentID.assign(line + 1, lineLen - 1);
        }
        else
            currentSeq.append(line, lineLen);
    }

    if (!currentSeq.empty())
    {
        result.ids.push_back(std::move(currentID));
        result.sequences.push_back(std::move(currentSeq));
    }

    return result;
}

// A reference window. `id` points into ParseResult::ids and `seq` views
// into ParseResult::sequences, so windows no longer duplicate the contig
// bytes (previously every window owned a string copy, doubling host memory
// on large references); the ParseResult must outlive the windows.
struct SeqWindow
{
    const string *id;
    size_t start, end;
    string_view seq;
};

// Position-partitioning chunker for the mapper: chunks overlap by k-1
// BASES but step by (len - k + 1) POSITIONS, so — unlike chunkSequences,
// whose windows share their boundary position by design — every k-mer
// position of a contig appears in exactly one chunk. SeqWindow.start holds
// the chunk's contig-global base offset; chunkContig[i] the contig index.
// Contigs shorter than k produce no chunks (they have no positions).
void chunkForMapping(const vector<string> &ids, const vector<string> &sequences,
                     size_t chunkSize, size_t kmerSize,
                     vector<SeqWindow> &chunks, vector<size_t> &chunkContig)
{
    // chunkSize < kmerSize would make the step (len - kmerSize + 1)
    // non-positive: clamp so the loop always advances.
    chunkSize = max(chunkSize, kmerSize);
    chunks.clear();
    chunkContig.clear();
    for (size_t c = 0; c < sequences.size(); c++)
    {
        const string_view seq = sequences[c];
        for (size_t o = 0; o + kmerSize <= seq.size();)
        {
            const size_t len = min(chunkSize, seq.size() - o);
            chunks.push_back(SeqWindow{&ids[c], o, o + len, seq.substr(o, len)});
            chunkContig.push_back(c);
            o += len - kmerSize + 1;
        }
    }
}

void chunkSequences(const vector<string> &ids, const vector<string> &sequences, size_t chunkSize, size_t kmerSize,
                    vector<SeqWindow> &chunks)
{
    // Windows step by chunkSize - kmerSize: a window of k bases or fewer
    // would never advance (or wrap the step). The tools reject such a
    // window size up front; this guards every other caller.
    if (chunkSize <= kmerSize)
        throw invalid_argument("window size " + to_string(chunkSize) +
                               " must be larger than k (" + to_string(kmerSize) + ")");
    chunks.clear();
    for (size_t i = 0; i < sequences.size(); i++)
    {
        const string &id = ids[i];
        const string_view sequence = sequences[i];
        for (size_t j = 0; j < sequence.size(); j += chunkSize - kmerSize)
        {
            size_t end;
            string_view chunk;
            if (j + chunkSize > sequence.size())
            {
                end = sequence.size();
                chunk = sequence.substr(j);
            }
            else
            {
                end = j + chunkSize;
                chunk = sequence.substr(j, chunkSize);
            }
            chunks.push_back(SeqWindow{&id, j, end, chunk});
        }
    }
}