#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include "KmerDatabase.hpp"
#include "KmerPanel.hpp"
#include "GpuCli.hpp"
#include "CliInput.hpp"

using namespace std;
namespace fs = filesystem;

std::string getReferenceName(const std::string &filename)
{
    size_t pos = filename.find("_genome");
    if (pos != std::string::npos)
    {
        return filename.substr(0, pos);
    }
    else
    {
        return filename;
    }
}

void removeTrailingSlash(string &path)
{
    if (!path.empty() && path.back() == '/')
    {
        path.pop_back();
    }
}

// As in the original: the error goes to log.txt in the working directory
// and the next reference is processed. It is also said on stderr, and the
// exit status reports it -- a reference whose output is missing must not
// look like a successful run (job 68747: a device allocation failed,
// log.txt held the only trace, and the exit status was 0).
void logReferenceError(const string &refName, const std::exception &e)
{
    cerr << "Error processing reference " << refName << ": " << e.what()
         << " (also in log.txt)" << endl;
    ofstream logFile("log.txt", ios::app); // Open the log file in append mode
    if (logFile.is_open())
    {
        logFile << "Error processing reference: " << refName << endl;
        logFile << e.what() << endl;
        logFile.close();
    }
    else
    {
        cout << "Unable to open log file." << endl;
    }
}

// One database against every reference: the original tool's run.
// dbDir is the dataset folder; its name names the output files.
int runSingle(const string &dbDir, const vector<fs::path> &references,
              const string &referencePath, const string &resultsFolder, int windowSize)
{
    string sourcePath;
    if (!cli::resolveDatabase(dbDir, sourcePath))
        return 1;
    const string database = std::filesystem::path(dbDir).filename().string();
    cout << "****Database: " << database << endl;

    cout << "Using sourcePath: " << sourcePath << endl;
    cout << "Using referencePath: " << referencePath << endl;
    cout << "Using resultsFolder: " << resultsFolder << endl;

    auto start = chrono::high_resolution_clock::now();
    cout << "Loading KMC database from " << sourcePath << endl;
    KmerDatabase db(sourcePath);
    db.printKMCInfo();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Database loaded in " << elapsed.count() << " seconds" << endl;
    cout << "********************************************" << endl;
    if (!cli::windowFitsK(windowSize, db.getKmerSize()))
    {
        cli::reportWindowTooSmall(windowSize, db.getKmerSize());
        return 1;
    }

    /************************************************************/

    start = chrono::high_resolution_clock::now();
    if (!cli::prepareResultsFolder(resultsFolder))
        return 1;

    int failed = 0;
    for (const auto &ref : references)
    {
        string refPath = ref.string();
        string refName = ref.filename().string();
        refName = getReferenceName(refName);
        cout << "Processing reference: " << refName << endl;
        auto outPath = resultsFolder + "/" + database + "_v_" + refName + "_" + to_string(windowSize) + ".tsv";
        //check if file exists, if so skip
        if (fs::exists(outPath))
        {
            cout << "File already exists, skipping..." << endl;
            continue;
        }
        try
        {
            db.processReference(refPath, outPath, windowSize);
        }
        catch (const std::exception &e)
        {
            failed++;
            logReferenceError(refName, e);
            continue;
        }
    }

    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "Reference processed in " << elapsed.count() << " seconds" << endl;
    if (failed)
        cerr << failed << " reference(s) failed; see log.txt" << endl;

    /************************************************************/

    return failed ? 1 : 0;
}

// Panel mode: the given dataset folders against every reference, one
// reference pass per reference. Same file names, same skip-if-exists rule,
// same error logging as the single-database run.
int runPanel(vector<string> dbDirs, const vector<fs::path> &references,
             const string &referencePath, const string &resultsFolder, int windowSize)
{
    vector<string> prefixes, names;
    for (string &dir : dbDirs)
    {
        removeTrailingSlash(dir);
        const string name = std::filesystem::path(dir).filename().string();
        // The output file is named after the folder, so two databases in
        // folders of the same name would write the same file.
        if (std::find(names.begin(), names.end(), name) != names.end())
        {
            cerr << "Error: two panel databases are named " << name
                 << " (their outputs would share a file name)" << endl;
            return 1;
        }
        string prefix;
        if (!cli::resolveDatabase(dir, prefix))
            return 1;
        names.push_back(name);
        prefixes.push_back(prefix);
    }

    // k > 32: no panel path packs such k-mers (KmerPanel refuses them), so
    // the databases are scanned one at a time on the original string path,
    // writing the same files a panel or N single runs would. Every
    // member's k is read first (headers only): a panel shares one k, which
    // KmerPanel also checks, but the string path below would not.
    try
    {
        const uint32_t k = KmcRaIndex::readHeader(prefixes[0]).kmerLength;
        for (size_t i = 1; i < prefixes.size(); i++)
        {
            const uint32_t ki = KmcRaIndex::readHeader(prefixes[i]).kmerLength;
            if (ki != k)
            {
                cerr << "Error: KmerPanel: " << names[i] << " has k=" << ki << " but " << names[0]
                     << " has k=" << k << "; a panel shares one k" << endl;
                return 1;
            }
        }
        if (KmerDatabase::needsStringPath(k))
        {
            cout << "[panel] k=" << k << " > 32 has no panel path; scanning the " << dbDirs.size()
                 << " databases one at a time on the original string path" << endl;
            int status = 0;
            for (const string &dir : dbDirs)
                status |= runSingle(dir, references, referencePath, resultsFolder, windowSize);
            return status;
        }
    }
    catch (const std::exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    cout << "****Panel: " << dbDirs.size() << " databases" << endl;
    for (size_t i = 0; i < prefixes.size(); i++)
        cout << "  " << names[i] << " (" << prefixes[i] << ")" << endl;
    cout << "Using referencePath: " << referencePath << endl;
    cout << "Using resultsFolder: " << resultsFolder << endl;

    auto start = chrono::high_resolution_clock::now();
    std::unique_ptr<KmerPanel> panel;
    try
    {
        panel = std::make_unique<KmerPanel>(prefixes, names, panelOptions().resident);
    }
    catch (const std::exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Database headers loaded in " << elapsed.count() << " seconds (k="
         << panel->kmerSize() << ")" << endl;
    cout << "********************************************" << endl;
    if (!cli::windowFitsK(windowSize, panel->kmerSize()))
    {
        cli::reportWindowTooSmall(windowSize, panel->kmerSize());
        return 1;
    }

    start = chrono::high_resolution_clock::now();
    if (!cli::prepareResultsFolder(resultsFolder))
        return 1;

    int failed = 0;
    for (const auto &ref : references)
    {
        string refPath = ref.string();
        string refName = getReferenceName(ref.filename().string());
        cout << "Processing reference: " << refName << endl;
        vector<string> outPaths(panel->size());
        size_t pending = 0;
        for (size_t i = 0; i < panel->size(); i++)
        {
            outPaths[i] = resultsFolder + "/" + panel->name(i) + "_v_" + refName + "_" +
                          to_string(windowSize) + ".tsv";
            if (fs::exists(outPaths[i]))
            {
                cout << "File already exists for " << panel->name(i) << ", skipping..." << endl;
                outPaths[i].clear();
            }
            else
                pending++;
        }
        if (!pending)
            continue;
        try
        {
            panel->processReference(refPath, outPaths, windowSize);
        }
        catch (const std::exception &e)
        {
            failed++;
            logReferenceError(refName, e);
            continue;
        }
    }
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "Reference processed in " << elapsed.count() << " seconds" << endl;
    if (failed)
        cerr << failed << " reference(s) failed; see log.txt" << endl;
    return failed ? 1 : 0;
}

int run(int argc, char *argv[])
{
    string sourcePath, referencePath, resultsFolder;
    int windowSize;

    // Usage text -- printed, as in the original tool, whenever the command
    // line is not accepted (too few arguments, an unknown option, a bad
    // value; exit status 1), and for -h / --help (exit status 0).
    auto printUsage = [&]()
    {
        cout << "\nFastIBS - IBS Distance Calculator\n"
             << "----------------------------------\n"
             << "Usage:\n"
             << "  " << argv[0] << " <sourcePath> <referencePath> <resultsFolder> <windowSize> [options]\n\n"
             << "Arguments:\n"
             << "  <sourcePath>     Path to folder with KMC dataset\n"
             << "                   e.g., /mnt/data/kmc_sets/BW_01002\n\n"
             << "  <referencePath>  Path to folder with reference genomes (FASTA)\n"
             << "                   e.g., /mnt/data/reference\n\n"
             << "  <resultsFolder>  Path to folder for storing output results\n"
             << "                   e.g., /mnt/data/FastIBS_runs\n\n"
             << "  <windowSize>     Length of the sequence window for IBS calculation\n"
             << "                   (integer larger than k, e.g., 50000)\n\n"
             << gpuOptionsUsage() << "\n"
             << panelOptionsUsage() << "\n"
             << "Notes:\n"
             << "  - All folders should be located on a mounted data volume.\n"
             << "  - Reference files: *.fasta, *.fa or *.fna, optionally gzip-compressed\n"
             << "    (*.fasta.gz, ...); other files in <referencePath> are skipped.\n"
             << "  - k <= 32 takes the accelerated paths; a database with k > 32 runs the\n"
             << "    original string path (GPU and --probe options are then ignored).\n"
             << "  - Exit status: 0 when every reference was processed or already had its\n"
             << "    output; 1 on a bad command line or input, or a failed reference\n"
             << "    (logged to log.txt).\n\n"
             << "Output:\n"
             << "  A tab-delimited file summarizing IBS distance metrics for each window.\n"
             << "  Columns: seqname, start, end, total_kmers, observed_kmers, variations, kmer_distance\n\n";
    };
    if (cli::versionRequested(argc, argv))
    {
        cout << cli::versionLine("fastibs") << endl;
        return 0;
    }
    if (cli::helpRequested(argc, argv))
    {
        printUsage();
        return 0;
    }
    if (argc < 5)
    {
        printUsage();
        return 1;
    }
    sourcePath = argv[1];
    referencePath = argv[2];
    resultsFolder = argv[3];
    if (!cli::parseWindowSize(argv[4], windowSize))
    {
        cerr << "Invalid <windowSize>: " << argv[4] << " (expected a positive integer)" << endl;
        printUsage();
        return 1;
    }

    // Everything after the four positional arguments is optional (GPU knobs,
    // panel databases).
    if (!parseGpuOptions(argc, argv, 5, true))
    {
        printUsage();
        return 1;
    }

    removeTrailingSlash(sourcePath);
    removeTrailingSlash(referencePath);
    removeTrailingSlash(resultsFolder);

    // Panel mode: <sourcePath> plus every --db / --db-list database, in one
    // reference pass (docs/MULTIDB_SCAN.md). Without those options the
    // single-database run below is untouched.
    vector<string> dbDirs = panelOptions().dbs;
    if (!panelOptions().listFile.empty())
    {
        ifstream list(panelOptions().listFile);
        if (!list)
        {
            cerr << "Error: cannot open --db-list file: " << panelOptions().listFile << endl;
            return 1;
        }
        string line;
        while (getline(list, line))
        {
            const size_t a = line.find_first_not_of(" \t\r");
            if (a == string::npos || line[a] == '#')
                continue;
            const size_t b = line.find_last_not_of(" \t\r");
            dbDirs.push_back(line.substr(a, b - a + 1));
        }
    }

    vector<fs::path> references;
    if (!cli::listReferences(referencePath, references))
        return 1;

    if (!dbDirs.empty())
    {
        dbDirs.insert(dbDirs.begin(), sourcePath);
        return runPanel(dbDirs, references, referencePath, resultsFolder, windowSize);
    }
    return runSingle(sourcePath, references, referencePath, resultsFolder, windowSize);
}

int main(int argc, char *argv[])
{
    // Anything the checks above did not anticipate (a folder that turns
    // unreadable, an allocation failure) ends in one error line and exit
    // status 1, not an uncaught exception's abort.
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
