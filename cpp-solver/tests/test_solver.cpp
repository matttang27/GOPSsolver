#include <gtest/gtest.h>
#include "solver.h"

TEST(SolverTest, TestExample) {
    Solver solver;
    // Add test cases to validate solver functionality
    EXPECT_EQ(solver.solve(1, 2), 3); // Example test case
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}