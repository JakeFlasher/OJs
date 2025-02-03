#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    // Use fast I/O for competitive programming.
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int N, K;
    cin >> N >> K;
    
    // Use long long to avoid overflow (since A[i] can be up to 1e9 and K up to 1e5).
    vector<long long> A(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    
    // Sort the prices in non-decreasing order.
    sort(A.begin(), A.end());
    
    // Sum the base prices of the K cheapest items.
    long long totalCost = 0;
    for (int i = 0; i < K; i++) {
        totalCost += A[i];
    }
    
    // Add the extra cost for waiting days: 0 + 1 + ... + (K-1).
    totalCost += static_cast<long long>(K) * (K - 1) / 2;
    
    cout << totalCost << "\n";
    
    return 0;
}
