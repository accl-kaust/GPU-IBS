#include <iostream>
#include <string>
#include <filesystem>
#include "KmerDatabase.hpp"
#include "GpuCli.hpp"
#include "CliInput.hpp"

using namespace std;
namespace fs = std::filesystem;


void removeTrailingSlash(string &path)
{
    if (!path.empty() && path.back() == '/')
    {
        path.pop_back();
    }
}

int run(int argc, char *argv[])
{
    string sourcePath, referencePath, resultsFolder, database;

    // Usage text -- printed, as in the original tool, whenever the command
    // line is not accepted (too few arguments, an unknown option, a bad
    // value; exit status 1), and for -h / --help (exit status 0).
    auto printUsage = [&]()
    {
        cout << "\nFastIBS - Reference Mapping Tool\n"
             << "---------------------------------\n"
             << "Usage:\n"
             << "  " << argv[0] << " <sourcePath> <referencePath> <resultsFolder> [options]\n\n"
             << "Arguments:\n"
             << "  <sourcePath>     Path to folder containing KMC database files\n"
             << "                   (e.g., /mnt/data/kmc_sets/<dataset_name>)\n\n"
             << "  <referencePath>  Path to folder containing reference genomes in FASTA format\n"
             << "                   (e.g., /mnt/data/reference)\n\n"
             << "  <resultsFolder>  Destination folder for writing mapping result files\n"
             << "                   (e.g., /mnt/data/FastIBS_runs)\n\n"
             << gpuOptionsUsage() << "\n"
             << "Notes:\n"
             << "  - All input folders should reside on a mounted data volume.\n"
             << "  - The tool scans <referencePath> for *.fasta, *.fa and *.fna files, optionally\n"
             << "    gzip-compressed (*.fasta.gz, ...), and processes them against the KMC base;\n"
             << "    other files are skipped.\n"
             << "  - Output filenames follow the format: <sourcePath folder>_v_<reference_stem>.txt\n"
             << "    (the stem without .gz and the FASTA extension)\n"
             << "  - Existing output files will be skipped.\n"
             << "  - Errors during processing are logged to log.txt.\n"
             << "  - k <= 32 takes the accelerated paths; a database with k > 32 runs the\n"
             << "    original string path (GPU and --probe options are then ignored).\n"
             << "  - Exit status: 0 when every reference was processed or already had its\n"
             << "    output; 1 on a bad command line or input, or a failed reference.\n\n"
             << "Example:\n"
             << "  " << argv[0] << " /mnt/data/kmc_sets/sample1 /mnt/data/reference /mnt/data/FastIBS_runs\n\n"
             << "Output: This tool computes a K-mer mapping for the given references, where each nucleotide position in the reference "
             << "sequences is associated with a count of how many K-mers (of a fixed size, defined by the kmerSize of the KMC source) "
             << "overlap that position and exist in the source KMC database.\n";
    };
    if (cli::versionRequested(argc, argv))
    {
        cout << cli::versionLine("fastibsmapper") << endl;
        return 0;
    }
    if (cli::helpRequested(argc, argv))
    {
        printUsage();
        return 0;
    }
    if (argc < 4)
    {
        printUsage();
        return 1;
    }
    sourcePath = argv[1];
    referencePath = argv[2];
    resultsFolder = argv[3];
    // Everything after the three positional arguments is optional (GPU knobs).
    if (!parseGpuOptions(argc, argv, 4))
    {
        printUsage();
        return 1;
    }

    removeTrailingSlash(sourcePath);
    removeTrailingSlash(referencePath);
    removeTrailingSlash(resultsFolder);

    const string sourceDir = sourcePath;
    if (!cli::resolveDatabase(sourceDir, sourcePath))
        return 1;
    vector<fs::path> references;
    if (!cli::listReferences(referencePath, references))
        return 1;

    database = std::filesystem::path(sourceDir).filename().string();
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

    /************************************************************/

    start = chrono::high_resolution_clock::now();
    if (!cli::prepareResultsFolder(resultsFolder))
        return 1;

    int failed = 0;
    for (const auto &ref : references)
    {
        string refPath = ref.string();
        // x.fasta -> x, as the original named it; x.fasta.gz -> x as well
        string refName = fs::path(cli::stripGz(ref.filename().string())).stem().string();
        cout << "Processing reference: " << refName << endl;
        auto outPath = resultsFolder + "/" + database + "_v_" + refName + ".txt";
        //check if file exists, if so skip
        if (fs::exists(outPath))
        {
            cout << "File already exists, skipping..." << endl;
            continue;
        }
        try
        {
            db.produceMapping(refPath, outPath);
        }
        catch (const std::exception &e)
        {
            // As in the original: the error goes to log.txt in the working
            // directory and the next reference is processed. It is also said
            // on stderr, and the exit status reports it -- a reference whose
            // output is missing must not look like a successful run (job
            // 68747: a device allocation failed, log.txt held the only
            // trace, and the exit status was 0).
            failed++;
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

int main(int argc, char *argv[])
{
    // Anything the input checks did not anticipate ends in one error line
    // and exit status 1, not an uncaught exception's abort.
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
