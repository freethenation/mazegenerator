#include "maze.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include "depthfirstsearch.h"
#include "cellborder.h"

Maze::Maze(int vertices, int startvertex, int endvertex)
    : vertices_(vertices), startvertex_(startvertex), endvertex_(endvertex) {}

void Maze::InitialiseGraph() {
  adjacencylist_.clear();
  adjacencylist_.resize(vertices_);
}

void Maze::GenerateMaze(SpanningtreeAlgorithm* algorithm) {
  auto spanningtree = algorithm->SpanningTree(vertices_, adjacencylist_);
  Solve(spanningtree);
  RemoveBorders(spanningtree);
}

void Maze::Solve(const std::vector<std::pair<int, int>>& edges) {
  Graph spanningtreegraph(vertices_);
  for (const auto& [u, v] : edges) {
    spanningtreegraph[u].push_back(
        *std::find_if(adjacencylist_[u].begin(), adjacencylist_[u].end(),
                      [v = v](const Edge& e) { return std::get<0>(e) == v; }));
    spanningtreegraph[v].push_back(
        *std::find_if(adjacencylist_[v].begin(), adjacencylist_[v].end(),
                      [u = u](const Edge& e) { return std::get<0>(e) == u; }));
  }

  DepthFirstSearch D;
  auto parent = D.Solve(vertices_, spanningtreegraph, startvertex_);
  solution_ = Graph(vertices_);
  for(int u = endvertex_; parent[u]!=u; u=parent[u]) {
    solution_[u].push_back(*std::find_if(
        spanningtreegraph[u].begin(), spanningtreegraph[u].end(),
        [u, &parent](const Edge& e) { return std::get<0>(e) == parent[u]; }));
  }
}

void Maze::RemoveBorders(const std::vector<std::pair<int, int>>& edges) {
  for (const auto& [u, v] : edges) {
    adjacencylist_[u].erase(
        std::find_if(adjacencylist_[u].begin(), adjacencylist_[u].end(),
                     [v = v](const Edge& e) { return std::get<0>(e) == v; }));
    adjacencylist_[v].erase(
        std::find_if(adjacencylist_[v].begin(), adjacencylist_[v].end(),
                     [u = u](const Edge& e) { return std::get<0>(e) == u; }));
  }
}

std::pair<double, double> Maze::GetVertexCenter(int vertex_index) const {
  if (vertex_index < 0 || vertex_index >= vertices_) {
    return {0.0, 0.0};
  }

  double sum_x = 0.0, sum_y = 0.0;
  int count = 0;

  // Try solution_ graph first (has path edges)
  for (const auto& edge : solution_[vertex_index]) {
    int neighbor = std::get<0>(edge);
    if (neighbor == -1) continue;

    auto border = std::get<1>(edge);

    if (auto line_border = std::dynamic_pointer_cast<LineBorder>(border)) {
      sum_x += line_border->x1_ + line_border->x2_;
      sum_y += line_border->y1_ + line_border->y2_;
      count += 2;
    } else if (auto arc_border = std::dynamic_pointer_cast<ArcBorder>(border)) {
      double x1 = arc_border->cx_ + arc_border->r_ * cos(arc_border->theta1_);
      double y1 = arc_border->cy_ + arc_border->r_ * sin(arc_border->theta1_);
      double x2 = arc_border->cx_ + arc_border->r_ * cos(arc_border->theta2_);
      double y2 = arc_border->cy_ + arc_border->r_ * sin(arc_border->theta2_);
      sum_x += x1 + x2;
      sum_y += y1 + y2;
      count += 2;
    }
  }

  // Fall back to adjacencylist_ if solution_ doesn't have edges
  if (count == 0) {
    for (const auto& edge : adjacencylist_[vertex_index]) {
      int neighbor = std::get<0>(edge);
      if (neighbor == -1) continue;

      auto border = std::get<1>(edge);

      if (auto line_border = std::dynamic_pointer_cast<LineBorder>(border)) {
        sum_x += line_border->x1_ + line_border->x2_;
        sum_y += line_border->y1_ + line_border->y2_;
        count += 2;
      } else if (auto arc_border = std::dynamic_pointer_cast<ArcBorder>(border)) {
        double x1 = arc_border->cx_ + arc_border->r_ * cos(arc_border->theta1_);
        double y1 = arc_border->cy_ + arc_border->r_ * sin(arc_border->theta1_);
        double x2 = arc_border->cx_ + arc_border->r_ * cos(arc_border->theta2_);
        double y2 = arc_border->cy_ + arc_border->r_ * sin(arc_border->theta2_);
        sum_x += x1 + x2;
        sum_y += y1 + y2;
        count += 2;
      }
    }
  }

  if (count == 0) {
    return {0.0, 0.0};
  }

  return {sum_x / count, sum_y / count};
}

std::vector<int> Maze::GetSolutionPath() const {
  std::vector<int> path;

  int current = endvertex_;
  while (current != startvertex_) {
    path.push_back(current);

    if (solution_[current].empty()) {
      return std::vector<int>();
    }

    current = std::get<0>(solution_[current][0]);
  }
  path.push_back(startvertex_);

  std::reverse(path.begin(), path.end());
  return path;
}

void Maze::PrintMazeGnuplot(const std::string& outputprefix, bool solution) const {
  std::ofstream gnuplotfile(outputprefix + ".plt");
  if (!gnuplotfile) {
    std::cerr << "Error opening " << outputprefix << ".plt for writing.\n";
    std::cerr << "Terminating.";
    exit(1);
  }

  gnuplotfile << "unset border\n";
  gnuplotfile << "unset tics\n";
  gnuplotfile << "set samples 15\n";
  gnuplotfile << "set lmargin at screen 0\n";
  gnuplotfile << "set rmargin at screen 1\n";
  gnuplotfile << "set bmargin at screen 0\n";
  gnuplotfile << "set tmargin at screen 1\n";

  double xmin, ymin, xmax, ymax;
  std::tie(xmin, ymin, xmax, ymax) = GetCoordinateBounds();
  double padding = 0.15;  // small padding to avoid clipping outer lines
  gnuplotfile << "set xrange[" << xmin - padding << ":" << xmax + padding << "]\n";
  gnuplotfile << "set yrange[" << ymin - padding << ":" << ymax + padding << "]\n";

  int xresolution = (xmax - xmin + 2 * padding) * 30,
      yresolution = (ymax - ymin + 2 * padding) * 30;
  gnuplotfile << "set term pngcairo enhanced size " << xresolution << ","
              << yresolution << "\n";

  gnuplotfile << "set output '" << outputprefix << ".png'\n";
  gnuplotfile << "set multiplot\n";
  for (int i = 0; i < vertices_; ++i) {
    for (const auto& edge : adjacencylist_[i]) {
      if (std::get<0>(edge) < i)
        gnuplotfile << std::get<1>(edge)->GnuplotPrintString("black") << "\n";
    }
  }

  if (solution) {
    auto path = GetSolutionPath();
    if (path.size() >= 2) {
      // Draw solution path (thicker to ensure clear path after color removal)
      for (size_t i = 0; i < path.size() - 1; ++i) {
        auto [x1, y1] = GetVertexCenter(path[i]);
        auto [x2, y2] = GetVertexCenter(path[i + 1]);
        gnuplotfile << "set arrow from " << x1 << "," << y1
                    << " to " << x2 << "," << y2
                    << " nohead lc'red' lw 12\n";
      }

      // Add Start and End symbols (circles)
      auto [start_x, start_y] = GetVertexCenter(path[0]);
      auto [end_x, end_y] = GetVertexCenter(path[path.size() - 1]);

      // Draw filled circle for START (green) - front layer ensures it's on top of red line
      gnuplotfile << "set object circle at " << start_x << "," << start_y
                  << " size 0.6 fc rgb 'green' fillstyle solid front\n";

      // Draw filled circle for END (blue) - front layer ensures it's on top of red line
      gnuplotfile << "set object circle at " << end_x << "," << end_y
                  << " size 0.6 fc rgb 'blue' fillstyle solid front\n";
    }
  }

  gnuplotfile << "plot 1/0 notitle\n";
  gnuplotfile << "unset multiplot\n";
  gnuplotfile << "set output\n";
}

void Maze::PrintMazeSVG(const std::string& outputprefix, bool solution) const {
  std::ofstream svgfile(outputprefix + ".svg");
  if (!svgfile) {
    std::cerr << "Error opening " << outputprefix << ".svg for writing.\n";
    std::cerr << "Terminating.";
    exit(1);
  }
  double xmin, ymin, xmax, ymax;
  std::tie(xmin, ymin, xmax, ymax) = GetCoordinateBounds();
  int xresolution = (xmax - xmin + 2) * 30,
      yresolution = (ymax - ymin + 2) * 30;

  svgfile << "<svg width=\"" << xresolution << "\" height=\"" << yresolution
          << "\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;
  svgfile << "<g transform=\"translate(" << (1 - xmin) * 30 << ","
          << yresolution - (1 - ymin) * 30 << ") scale(1,-1)\">" << std::endl;
  svgfile << "<rect x=\"" << (xmin - 1) * 30 << "\" y=\"" << (ymin - 1) * 30
          << "\" width=\"" << xresolution << "\" height=\"" << yresolution
          << "\" fill=\"white\"/>" << std::endl;

  for (int i = 0; i < vertices_; ++i) {
    for (const auto& edge : adjacencylist_[i]) {
      if (std::get<0>(edge) < i) {
        svgfile << std::get<1>(edge)->SVGPrintString("black") << "\n";
      }
    }
  }

  if (solution) {
    auto path = GetSolutionPath();
    if (!path.empty()) {
      svgfile << "<polyline points=\"";
      for (size_t i = 0; i < path.size(); ++i) {
        auto [x, y] = GetVertexCenter(path[i]);
        svgfile << x * 30 << "," << y * 30;
        if (i < path.size() - 1) svgfile << " ";
      }
      svgfile << "\" stroke=\"red\" stroke-width=\"2\" "
              << "fill=\"none\" stroke-linecap=\"round\" "
              << "stroke-linejoin=\"round\"/>\n";
    }
  }

  svgfile << "</g>" << std::endl;
  svgfile << "</svg>" << std::endl;
}
