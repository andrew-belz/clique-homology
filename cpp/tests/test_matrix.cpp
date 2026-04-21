#include <iostream>
#include <cassert>
#include <vector>
#include <utility>
#include "../src/internal/matrix.hpp"

void test_bit_boundaries() {
    std::cout << "Running Test 1: 64-bit Chunk Boundaries..." << std::endl;
    // 2 rows, 130 columns (Forces the matrix to use 3 integers per row)
    BinaryMatrix mat(2, 130); 
    
    // Test the edges of the first and second chunks
    mat.one(0, 63); // Very last bit of chunk 0
    mat.one(0, 64); // Very first bit of chunk 1
    mat.one(1, 129); // Very last column in the matrix

    assert(mat(0, 63) == 1);
    assert(mat(0, 64) == 1);
    assert(mat(0, 65) == 0); // Ensure neighbors aren't bleeding
    assert(mat(1, 129) == 1);
    
    // Test zeroing
    mat.zero(0, 64);
    assert(mat(0, 64) == 0);
}

void test_gf2_cancellation() {
    std::cout << "Running Test 2: GF(2) Linear Dependence..." << std::endl;
    BinaryMatrix mat(3, 3);
    
    // Row 0: [1, 1, 0]
    mat.one(0, 0); mat.one(0, 1);
    // Row 1: [1, 0, 1]
    mat.one(1, 0); mat.one(1, 2);
    // Row 2: [0, 1, 1]
    mat.one(2, 1); mat.one(2, 2);

    mat.print(true);

    // Because Row 0 ^ Row 1 == Row 2, 
    // Row 2 is linearly dependent and should completely zero out
    auto [rank, nullity] = mat.row_reduce();
    
    assert(rank == 2);
    assert(nullity == 1); // 3 columns - 2 rank = 1
}

void test_empty_and_identity() {
    std::cout << "Running Test 3: Zero and Identity Matrices..." << std::endl;
    
    BinaryMatrix zero_mat(5, 5);
    auto [z_rank, z_null] = zero_mat.row_reduce();
    assert(z_rank == 0);
    assert(z_null == 5);

    zero_mat.print(false);

    BinaryMatrix id_mat(4, 4);
    for (size_t i = 0; i < 4; i++) id_mat.one(i, i);
    auto [id_rank, id_null] = id_mat.row_reduce();
    assert(id_rank == 4);
    assert(id_null == 0);
}

void test_cache_invalidation() {
    std::cout << "Running Test 4: Cache Invalidation..." << std::endl;
    BinaryMatrix mat(3, 3);
    
    // Start with a zero matrix
    auto [rank1, null1] = mat.row_reduce();
    assert(rank1 == 0);

    // Modify the matrix (this should flip row_reduced to false)
    mat.one(0, 0);
    mat.one(1, 1);
    mat.one(2, 2);

    // Re-reduce. If the cache bug is there, it will incorrectly return 0
    auto [rank2, null2] = mat.row_reduce();
    assert(rank2 == 3); 
    assert(null2 == 0);
}

void test_out_of_bounds() {
    std::cout << "Running Test 5: Out of Bounds Exceptions..." << std::endl;
    BinaryMatrix mat(2, 2);
    bool caught = false;
    
    try {
        mat.one(5, 5); // Should throw
    } catch (const std::out_of_range& e) {
        caught = true;
    }
    assert(caught == true);
}

void test_vec() {
    // test if it can properly print a vector or not
    std::cout << "Running Test 6: Vectors..." << std::endl;

    BinaryMatrix row_vec(1, 5);
    BinaryMatrix col_vec(5, 1);

    row_vec.one(0, 2);
    col_vec.one(3, 0);

    row_vec.print(false);
    col_vec.print(false);

    // make sure that row reducing doesn't have trouble with 1 dimensional vectors
    auto [rank1, nullity1] = row_vec.row_reduce();
    auto [rank2, nullity2] = col_vec.row_reduce();

    assert(rank1 == 1);
    assert(nullity1 == 4);

    assert(rank2 == 1);
    assert(nullity2 == 0);

    assert(row_vec*col_vec == BinaryMatrix(1, 1));

}

void test_mat_ops() {
    std::cout << "Running Test 7: Matrix Operations..." << std::endl;
    BinaryMatrix A(2, 2);
    BinaryMatrix B(2, 2);
    BinaryMatrix C(2, 2);

    // make sure == works

    assert(A == B);
    assert(B == C);

    // identity matrices
    A.one(0, 0);
    A.one(1, 1);

    B.one(0, 0);
    B.one(1, 1);

    // adding should give the zero matrix
    assert(A + B == C);

    // can operate on oneself
    assert(A + A == C);

    // multiplying should give back the identity
    assert(A * B == A && A * B == B);
    assert(A * A == A);

    BinaryMatrix row(1, 2);
    row.one(0, 0);

    // does my row extraction logic work properly
    assert(A.row_i(0) == row);

    // test boolean comparison for 1x1 matrices
    BinaryMatrix one_by_one_true(1, 1);
    one_by_one_true.one(0, 0);
    BinaryMatrix one_by_one_false(1, 1);

    assert(one_by_one_true == 1);
    assert(one_by_one_false == 0);
    assert(!(one_by_one_true == 0));
    assert(!(one_by_one_false == 1));

    // test the other way around
    assert(1 == one_by_one_true);
    assert(0 == one_by_one_false);
    assert(!(0 == one_by_one_true));
    assert(!(1 == one_by_one_false));

}


int main() {
    std::cout << "--- Starting BinaryMatrix Test Suite ---" << std::endl;
    
    test_bit_boundaries();
    test_gf2_cancellation();
    test_empty_and_identity();
    test_cache_invalidation();
    test_out_of_bounds();
    test_vec();
    test_mat_ops();

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Passed." << std::endl;
    
    return 0;
}