$(document).ready(function () {

    $('.failure-code').tooltip({container: 'body'});
    $('.td-job-id').tooltip({container: 'body'});

    // select drop-down box
    $('select').selectpicker();

    // close update message
    $("#update-message-close").click(function () {
        $("#update-message").text("");
        $(this).css("display", "none");
    });

    // close search message
    $("#filter-message-close").click(function () {
        $("#sn-search").attr("value","");
        $("#filter-message").text("");
        $("#comment_text").text();
        $("#job-form").submit();
    });

    // submit the job-type select
    $('#type-select').change(function () {
        $("#job-form").submit();
    });

    // submit the time-field select
    $('#time-select').change(function () {
        $("#job-form").submit();
    });

    // submit the search by vehicle SN
    $("#search-btn").click(function () {
        $("#job-form").submit();
    });

    // close search message
    $("#statistics-filter-message-close").click(function () {
        $("#statistics-filter-message").text("");
        $("#statictics-form").submit();
    });

    // submit the job-type select
    $('#statistics-type-select').change(function () {
        $("#statictics-form").submit();
    });

    // submit the time-field select
    $('#statistics-time-select').change(function () {
        $("#statictics-form").submit();
    });

    // submit the search by vehicle SN
    $("#statistics-search-btn").click(function () {
        $("#statictics-form").submit();
    });


    // submit the aggregate-field select
    $('#aggregated-select').change(function () {
        $("#statictics-form").submit();
    });

    // Modal box of comment
    $('#identifier').modal()

    // Click the body
    $('body').click(function () {
        $("#comment_text").val("");
    });

    // Click the job-modal
    $(".job-modal").click(function () {
        var job_id = $(this).attr("id").split("-")[1];
        var action = $(this).text().trim().slice(2, 4);
        var en_action = "";
        if (action == "无效") {
            en_action = "Invalid"
        } else if (action == "有效") {
            en_action = "Valid"
        }
        $(this).append('<div id=' + job_id + ' class="job-id"></div><div id="job-action" class=' + en_action + '></div>');
        $("#myModalLabel").text("设置" + action);
        $(".comment-title").text("你确认把当前任务设置为" + action + "吗?");
    })

    // Click the cancel button
    $(".cancel_job").click(function () {
        $("#comment_text").val("");
        $('#myModal').modal("hide")
    })

    // Click the x button
    $(".close").click(function () {
        $("#comment_text").val("");
        $('#myModal').modal("hide")
    })

    // Click the submit button
    $(".comment_job").click(function () {
        var comment = $("#comment_text").val();
        if (comment == "") {
            $('#comments-error').text("不能为空!")
            $('#comments-error').css({"color": "red"}).show(300).delay(3000).hide(300);
        } else {
            var job_id = $(".job-id").attr("id");
            var action = $("#job-action").attr("class");
            $(".job-id").remove();
            $("#job-action").remove();
            var job_data = {"comment": comment, "job_id": job_id, "action": action};
            $.ajax({
                url: "http://usa-data.baidu.com:8001/api/v1/namespaces/default/services/http:admin-console-service:8000/proxy/submit_job",
                dataType: "json",
                type: "POST",
                data: job_data,
                beforeSend: function () {
                    $("#comment_text").attr("disabled", "disabled");
                },
                success: function (result) {
                    $("#comment_text").val("");
                    $("#comment_text").removeAttr("disabled");
                    $('#myModal').modal("hide");
                    var action = result["operation"]["action"]["type"];
                    var app_action = ""
                    if (action == "invalid") {
                        app_action = "设置有效"
                        action = "无效"
                    } else if (action == "valid") {
                        app_action = "设置无效"
                        action = "有效"
                    } else {
                        app_action = "错误"
                        action = "错误"
                    }
                    $("#button-" + job_id).text(app_action);
                    $("#flag-" + job_id).before('<tr style="display: none" class="operation_span">' +
                        '<td colspan="11" style="border-top: none">' +
                        "任务被" + result["operation"]["email"] + "在" + result["operation"]["time"] +
                        "设置为" + action + "。备注是：" + result["operation"]["comments"] + "</td></tr>");
                    var is_valid = result["operation"]["is_valid"].toString()
                    is_valid = is_valid.substring(0, 1).toUpperCase() + is_valid.substring(1)
                    $("#is-valid-" + job_id).text(is_valid);
                    $('#update-message').text("任务（序号："+job_id+"）被设置为"+action);
                    $("#update-message-close").css("display", "inline");
                    $("#expand-" + job_id).css("display", "none");
                    $("#collapse-" + job_id).css("display", "inline");
                    $("#button-" + job_id).parent().parent().nextUntil(".flag").css("display", "table-row");
                },
                error: function (xhr, textStatus, errorThrown) {
                    $('#update-message').text("任务（序号：" + job_id + "）提交失败");
                    $("#update-message-close").css("display", "inline");
                }
            })
        }
    })

    // Click the expand
    $(".job_expand").click(function () {
        $(this).css("display", "none");
        $(this).next().css("display", "inline");
        $(this).parent().parent().nextUntil(".flag").css("display", "table-row")
    })

    // Click the collapse
    $(".job_collapse").click(function () {
        $(this).css("display", "none");
        $(this).prev().css("display", "inline");
        $(this).parent().parent().nextUntil(".flag").css("display", "none")
    })
})
