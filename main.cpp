#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Quad = std::vector<cv::Point>;

// 求四边形中心。
cv::Point2d centerOf(const Quad& q)
{
    cv::Point2d c(0, 0);
    for (const auto& p : q) {
        c.x += p.x;
        c.y += p.y;
    }
    return c * 0.25;
}

// 在图片坐标中按顺时针排列。
// 起点取最上方的点；同高时取最左方的点。
Quad orderCorners(Quad q)
{
    const auto c = centerOf(q);

    std::sort(q.begin(), q.end(), [&](const auto& a, const auto& b) {
        return std::atan2(a.y - c.y, a.x - c.x)
             < std::atan2(b.y - c.y, b.x - c.x);
    });

    auto first = std::min_element(
        q.begin(), q.end(), [](const auto& a, const auto& b) {
            if (a.y != b.y) return a.y < b.y;
            return a.x < b.x;
        });

    std::rotate(q.begin(), first, q.end());
    return q;
}

// 判断四个角是否都接近直角。
bool hasRightAngles(const Quad& q)
{
    for (int i = 0; i < 4; ++i) {
        cv::Point2d u = q[(i + 3) % 4] - q[i];
        cv::Point2d v = q[(i + 1) % 4] - q[i];

        const double denominator = cv::norm(u) * cv::norm(v);
        if (denominator < 1e-6) return false;

        const double cosine = std::abs(u.dot(v) / denominator);
        if (cosine > 0.20) return false;
    }
    return true;
}

// 比较四组角点，识别非常接近的重复候选。
// 同时尝试不同起点和方向，避免排序差异影响比较。
bool isDuplicate(const Quad& a, const Quad& b)
{
    double best = 1e30;

    for (int direction : {-1, 1}) {
        for (int shift = 0; shift < 4; ++shift) {
            double distance = 0;

            for (int i = 0; i < 4; ++i) {
                const int j = (shift + direction * i + 8) % 4;
                distance += cv::norm(a[i] - b[j]);
            }

            best = std::min(best, distance / 4.0);
        }
    }

    const double tolerance = std::max(
        4.0, 0.02 * std::sqrt(std::abs(cv::contourArea(a))));

    return best < tolerance;
}

std::vector<Quad> detectRectangles(const cv::Mat& image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 同时检查灰度图和三个颜色通道。
    std::vector<cv::Mat> planes;
    cv::split(image, planes);
    planes.push_back(gray);

    std::vector<Quad> candidates;

    for (const auto& plane : planes) {
        cv::Mat smooth;
        cv::GaussianBlur(plane, smooth, cv::Size(3, 3), 0);

        std::vector<cv::Mat> masks(3);

        cv::Canny(smooth, masks[0], 40, 120);
        cv::threshold(
            smooth, masks[1], 0, 255,
            cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::bitwise_not(masks[1], masks[2]);

        for (const auto& mask : masks) {
            std::vector<Quad> contours;
            cv::findContours(
                mask, contours,
                cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                const double perimeter = cv::arcLength(contour, true);

                Quad q;
                cv::approxPolyDP(contour, q, perimeter * 0.02, true);

                if (q.size() != 4) continue;
                if (!cv::isContourConvex(q)) continue;

                const double area = std::abs(cv::contourArea(q));

                // 起步版参数：过滤小噪声及整幅背景形成的外框。
                if (area < 64) continue;
                if (area > image.total() * 0.99) continue;
                if (!hasRightAngles(q)) continue;

                candidates.push_back(orderCorners(q));
            }
        }
    }

    // 优先保留面积较大的候选，再做简单去重。
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) {
            return std::abs(cv::contourArea(a))
                 > std::abs(cv::contourArea(b));
        });

    std::vector<Quad> result;

    for (const auto& q : candidates) {
        bool duplicate = false;

        for (const auto& accepted : result) {
            if (isDuplicate(q, accepted)) {
                duplicate = true;
                break;
            }
        }

        if (!duplicate) result.push_back(q);
    }

    // 多个矩形按中心从左到右排列，同列时从上到下。
    std::sort(result.begin(), result.end(),
        [](const auto& a, const auto& b) {
            const auto ca = centerOf(a);
            const auto cb = centerOf(b);
            if (ca.x != cb.x) return ca.x < cb.x;
            return ca.y < cb.y;
        });

    return result;
}

// 生成四张简单测试图片，方便先验证工程能运行。
void makeDemo()
{
    fs::create_directories("demo_input");

    cv::Mat blank(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::imwrite("demo_input/00_empty.png", blank);

    cv::Mat one = blank.clone();
    cv::rectangle(one, cv::Point(100, 100), cv::Point(300, 250),
                  cv::Scalar(40, 90, 180), cv::FILLED);
    cv::imwrite("demo_input/01_one.png", one);

    cv::Mat rotated = blank.clone();
    cv::RotatedRect box(
        cv::Point2f(300, 200), cv::Size2f(220, 110), 30);

    cv::Point2f points[4];
    box.points(points);

    Quad polygon;
    for (const auto& p : points)
        polygon.emplace_back(cvRound(p.x), cvRound(p.y));

    cv::fillConvexPoly(rotated, polygon, cv::Scalar(180, 80, 40));
    cv::imwrite("demo_input/02_rotated.png", rotated);

    cv::Mat multi = blank.clone();
    cv::rectangle(multi, cv::Point(70, 70), cv::Point(220, 180),
                  cv::Scalar(20, 20, 20), cv::FILLED);
    cv::rectangle(multi, cv::Point(360, 200), cv::Point(530, 330),
                  cv::Scalar(180, 60, 40), cv::FILLED);
    cv::imwrite("demo_input/03_multi.png", multi);
}

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "--make-demo") {
            makeDemo();
            return 0;
        }

        const fs::path input = argc > 1 ? argv[1] : "/input";
        const fs::path output = argc > 2 ? argv[2] : "/out";

        if (!fs::is_directory(input))
            throw std::runtime_error("Input directory does not exist.");

        if (fs::absolute(input).lexically_normal()
            == fs::absolute(output).lexically_normal())
            throw std::runtime_error("Input and output must be different.");

        fs::create_directories(output);

        std::vector<fs::path> files;

        for (const auto& entry : fs::directory_iterator(input)) {
            if (!entry.is_regular_file()) continue;

            std::string extension = entry.path().extension().string();
            std::transform(
                extension.begin(), extension.end(), extension.begin(),
                [](unsigned char c) { return std::tolower(c); });

            if (extension == ".png")
                files.push_back(entry.path());
        }

        std::sort(files.begin(), files.end(),
            [](const auto& a, const auto& b) {
                return a.filename().string() < b.filename().string();
            });

        bool failed = false;

        for (const auto& path : files) {
            cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);

            if (image.empty()) {
                std::cerr << "Cannot read: " << path << '\n';
                failed = true;
                continue;
            }

            const auto rectangles = detectRectangles(image);
            cv::Mat annotated = image.clone();

            if (!rectangles.empty()) {
                cv::polylines(
                    annotated, rectangles, true,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_8);
            }

            const fs::path destination = output / path.filename();

            if (!cv::imwrite(destination.string(), annotated))
                throw std::runtime_error("Cannot save output image.");

            std::cout << path.filename().string() << ": ";

            if (rectangles.empty()) {
                std::cout << "null";
            } else {
                for (std::size_t r = 0; r < rectangles.size(); ++r) {
                    if (r != 0) std::cout << " | ";
                    std::cout << "rect" << r + 1 << ' ';

                    for (int i = 0; i < 4; ++i) {
                        if (i != 0) std::cout << "; ";

                        const auto& p = rectangles[r][i];

                        // 从左上角坐标转换为左下角坐标。
                        const int x = p.x;
                        const int y = image.rows - 1 - p.y;

                        std::cout << '(' << x << ", " << y << ')';
                    }
                }
            }

            std::cout << '\n';
        }

        return failed ? 1 : 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
