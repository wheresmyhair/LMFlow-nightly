import unittest

import numpy as np
import torch

from lmflow.utils.protocol import DataProto


class TestDataProtoChunk(unittest.TestCase):
    def setUp(self):
        self.meta_info = {"source": "chunk-test"}
        self.data = DataProto.from_dict(
            tensors={"ids": torch.arange(5), "features": torch.arange(10).reshape(5, 2)},
            non_tensors={
                "labels": np.array(["a", "b", "c", "d", "e"], dtype=object),
                "indices": np.arange(5),
            },
            meta_info=self.meta_info,
        )

    def test_chunk_splits_divisible_data_equally(self):
        chunks = self.data[:4].chunk(2)

        self.assertEqual([len(chunk) for chunk in chunks], [2, 2])
        self.assertEqual([chunk.batch["ids"].tolist() for chunk in chunks], [[0, 1], [2, 3]])
        self.assertEqual([chunk.non_tensor_batch["labels"].tolist() for chunk in chunks], [["a", "b"], ["c", "d"]])

    def test_chunk_preserves_non_divisible_tensor_and_non_tensor_alignment(self):
        chunks = self.data.chunk(2)

        self.assertEqual([len(chunk) for chunk in chunks], [3, 2])
        self.assertEqual([chunk.batch["ids"].tolist() for chunk in chunks], [[0, 1, 2], [3, 4]])
        for chunk in chunks:
            np.testing.assert_array_equal(chunk.batch["ids"].numpy(), chunk.non_tensor_batch["indices"])
            self.assertIs(chunk.meta_info, self.meta_info)

        reconstructed_ids = torch.cat([chunk.batch["ids"] for chunk in chunks])
        reconstructed_labels = np.concatenate([chunk.non_tensor_batch["labels"] for chunk in chunks])
        torch.testing.assert_close(reconstructed_ids, self.data.batch["ids"])
        np.testing.assert_array_equal(reconstructed_labels, self.data.non_tensor_batch["labels"])

    def test_chunk_splits_non_tensor_only_data_with_the_same_semantics(self):
        data = DataProto.from_dict(
            non_tensors={"ids": np.arange(5), "labels": np.array(list("abcde"), dtype=object)},
            meta_info=self.meta_info,
        )

        chunks = data.chunk(3)

        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
        self.assertEqual([chunk.non_tensor_batch["ids"].tolist() for chunk in chunks], [[0, 1], [2, 3], [4]])
        self.assertTrue(all(chunk.batch is None for chunk in chunks))
        self.assertTrue(all(chunk.meta_info is self.meta_info for chunk in chunks))

    def test_chunk_returns_requested_empty_chunks_for_empty_data(self):
        data = DataProto(meta_info=self.meta_info)

        chunks = data.chunk(3)

        self.assertEqual(len(chunks), 3)
        self.assertEqual([len(chunk) for chunk in chunks], [0, 0, 0])
        self.assertTrue(all(chunk.meta_info is self.meta_info for chunk in chunks))

    def test_chunk_rejects_invalid_chunk_counts(self):
        for chunks in (0, -1):
            with self.subTest(chunks=chunks), self.assertRaisesRegex(ValueError, "greater than zero"):
                self.data.chunk(chunks)

        with self.assertRaisesRegex(ValueError, "cannot exceed DataProto size"):
            self.data.chunk(6)

        for chunks in (2.0, True):
            with self.subTest(chunks=chunks), self.assertRaisesRegex(TypeError, "must be an integer"):
                self.data.chunk(chunks)


if __name__ == "__main__":
    unittest.main()
