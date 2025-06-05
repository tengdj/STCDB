#ifndef SRC_INDEX_BPLUSTREE_H_
#define SRC_INDEX_BPLUSTREE_H_

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "../geometry/geometry.h"
using namespace std;


// Interval class representing a time range

struct Interval {
    uint start, end; // Time range [start, end]
    int value;       // Associated value

    // Default constructor
    Interval() : start(0), end(0), value(0) {}

    // Parameterized constructor
    Interval(uint s, uint e, int v) : start(s), end(e), value(v) {}

    Interval(const Interval& other)
            : start(other.start), end(other.end), value(other.value) {
    }

};

//struct Interval {
//    int start_time;
//    int end_time;
//
//    Interval(int start, int end) : start_time(start), end_time(end) {}
//
//    // Check if the interval intersects with another interval
//    bool intersects(const Interval& other) const {
//        return !(end_time < other.start_time || start_time > other.end_time);
//    }
//
//    // Print the interval
//    void print() const {
//        cout << "[" << start_time << ", " << end_time << "]";
//    }
//};
//
//// B+ Tree BNode structure
//template <typename ValueType>
//struct BNode {
//    bool is_leaf; // Whether the node is a leaf
//    vector<int> keys; // Keys used for routing in internal nodes (stores Interval start_time)
//    vector<BNode*> children; // Child pointers for internal nodes
//    vector<pair<Interval, ValueType>> entries; // Leaf node entries (Interval, ValueType)
//    BNode* next; // Pointer to the next leaf node for efficient range queries
//
//    BNode(bool leaf) : is_leaf(leaf), next(nullptr) {}
//};
//
//// B+ Tree class
//template <typename ValueType>
//class BPlusTree {
//private:
//    BNode<ValueType>* root; // Root node of the tree
//    int degree; // Degree of the B+ Tree (max number of children per node)
//
//    // Insert into a leaf node
//    void insertIntoLeaf(BNode<ValueType>* leaf, const Interval& interval, const ValueType& value) {
//        // Find the position to insert
//        auto it = lower_bound(leaf->keys.begin(), leaf->keys.end(), interval.start_time);
//        int index = distance(leaf->keys.begin(), it);
//
//        leaf->keys.insert(it, interval.start_time);
//        leaf->entries.insert(leaf->entries.begin() + index, {interval, value});
//    }
//
//    // Split a node
//    BNode<ValueType>* splitBNode(BNode<ValueType>* node) {
//        int mid = node->keys.size() / 2;
//
//        // Create a new node
//        BNode<ValueType>* new_node = new BNode<ValueType>(node->is_leaf);
//        new_node->keys.assign(node->keys.begin() + mid, node->keys.end());
//        node->keys.erase(node->keys.begin() + mid, node->keys.end());
//
//        if (node->is_leaf) {
//            new_node->entries.assign(node->entries.begin() + mid, node->entries.end());
//            node->entries.erase(node->entries.begin() + mid, node->entries.end());
//
//            // Update the linked list for leaf nodes
//            new_node->next = node->next;
//            node->next = new_node;
//        } else {
//            new_node->children.assign(node->children.begin() + mid, node->children.end());
//            node->children.erase(node->children.begin() + mid, node->children.end());
//        }
//
//        return new_node;
//    }
//
//    // Recursive insertion into the tree
//    void insert(BNode<ValueType>* node, const Interval& interval, const ValueType& value, int& up_key, BNode<ValueType>*& new_child) {
//        if (node->is_leaf) {
//            insertIntoLeaf(node, interval, value);
//
//            // Check if the node needs to be split
//            if (node->keys.size() >= degree) {
//                BNode<ValueType>* new_node = splitBNode(node);
//                up_key = new_node->keys[0];
//                new_child = new_node;
//            } else {
//                up_key = -1;
//                new_child = nullptr;
//            }
//        } else {
//            // Find the child node to traverse
//            auto it = upper_bound(node->keys.begin(), node->keys.end(), interval.start_time);
//            int index = distance(node->keys.begin(), it);
//            BNode<ValueType>* child = node->children[index];
//
//            int child_up_key;
//            BNode<ValueType>* child_new_node;
//            insert(child, interval, value, child_up_key, child_new_node);
//
//            if (child_new_node) {
//                node->keys.insert(node->keys.begin() + index, child_up_key);
//                node->children.insert(node->children.begin() + index + 1, child_new_node);
//
//                // Check if the node needs to be split
//                if (node->keys.size() >= degree) {
//                    BNode<ValueType>* new_node = splitBNode(node);
//                    up_key = new_node->keys[0];
//                    new_child = new_node;
//                } else {
//                    up_key = -1;
//                    new_child = nullptr;
//                }
//            } else {
//                up_key = -1;
//                new_child = nullptr;
//            }
//        }
//    }
//
//    // Search leaf nodes for intersecting intervals
//    void searchLeaf(BNode<ValueType>* leaf, const Interval& query, vector<pair<Interval, ValueType>>& result) const {
//        for (const auto& entry : leaf->entries) {
//            if (entry.first.intersects(query)) {
//                result.push_back(entry);
//            }
//        }
//    }
//
//public:
//    // Constructor
//    BPlusTree(int degree) : root(new BNode<ValueType>(true)), degree(degree) {}
//
//    // Insert an interval and value
//    void insert(const Interval& interval, const ValueType& value) {
//        int up_key;
//        BNode<ValueType>* new_child;
//        insert(root, interval, value, up_key, new_child);
//
//        if (new_child) {
//            // Create a new root
//            BNode<ValueType>* new_root = new BNode<ValueType>(false);
//            new_root->keys.push_back(up_key);
//            new_root->children.push_back(root);
//            new_root->children.push_back(new_child);
//            root = new_root;
//        }
//    }
//
//    // Search for intervals intersecting with the query interval
//    vector<pair<Interval, ValueType>> search(const Interval& query) const {
//        vector<pair<Interval, ValueType>> result;
//        BNode<ValueType>* node = root;
//
//        // Traverse down to a leaf node
//        while (!node->is_leaf) {
//            auto it = upper_bound(node->keys.begin(), node->keys.end(), query.start_time);
//            int index = distance(node->keys.begin(), it);
//            node = node->children[index];
//        }
//
//        // Search leaf nodes
//        while (node) {
//            searchLeaf(node, query, result);
//            node = node->next;
//        }
//
//        return result;
//    }
//};

#endif /* SRC_INDEX_BPLUSTREE_H_ */