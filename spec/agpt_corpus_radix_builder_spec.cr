require "./spec_helper"
require "file_utils"

private def read_radix_records(dir : String) : Array(MicroGPT::AGPT::RadixTrieReader::LoadedRecord)
  reader = MicroGPT::AGPT::RadixTrieReader.new(dir)
  records = [] of MicroGPT::AGPT::RadixTrieReader::LoadedRecord
  reader.each { |record| records << record }
  records
end

describe "AGPT corpus radix builder" do
  it "wraps tail lookahead into the corpus prefix without adding extra starts" do
    tokens = [0, 1, 2, 3] of Int32
    tmpdir = File.join(Dir.tempdir, "agpt_corpus_radix_wrap_#{Random.rand(UInt64)}")

    begin
      builder = MicroGPT::AGPT::CorpusRadixBuilder.new(
        corpus_tokens: tokens,
        vocab_size: 4,
        max_depth: 3,
        out_dir: tmpdir,
        corpus_hash: 0_u64,
        tokenizer_tag: "test",
        progress: false,
      )
      builder.build

      reader = MicroGPT::AGPT::RadixTrieReader.new(tmpdir)
      reader.corpus_token_count.should eq tokens.size

      by_edge = {} of Array(Int32) => Array({Int32, Int32})
      read_radix_records(tmpdir).each do |record|
        by_edge[record.edge_tokens] = record.counts
      end

      by_edge.size.should eq 4
      by_edge[[0, 1, 2]].should eq [{3, 1}]
      by_edge[[1, 2, 3]].should eq [{0, 1}]
      by_edge[[2, 3, 0]].should eq [{1, 1}]
      by_edge[[3, 0, 1]].should eq [{2, 1}]
    ensure
      FileUtils.rm_rf(tmpdir) if Dir.exists?(tmpdir)
    end
  end
end
